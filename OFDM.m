% OFDM Performance Analysis - MATLAB Simulation
%  Author: Shifa Parveen
%  Runs the FULL OFDM pipeline for BOTH AWGN and Rayleigh channels
%  automatically in a single run - no manual toggling needed.
%  Pipeline: QPSK modulation -> IFFT -> Cyclic Prefix -> Channel -> Receiver
%  -> BER vs Eb/N0, plus PAPR analysis.

clear;
clc; 
close all;

% PARAMETERS 
nBitsTotal  = 256*20;     % total bits (more data = smoother BER curve)
Nfft        = 64;         % number of subcarriers (FFT size)
Ncp         = 16;         % cyclic prefix length (25% of Nfft, typical)
EbN0_dB     = 0:2:20;     % range of Eb/N0 values to test (dB)
bitsPerSymbol = 2;        % QPSK
numTrials   = 30;         % independent trials per Eb/N0 point

% TRANSMITTER (built once, shared by both channels) 
bits = randi([0 1], 1, nBitsTotal);

bitPairs = reshape(bits, 2, [])';
symbols = zeros(1, size(bitPairs,1));

for k = 1:size(bitPairs,1)
    b = bitPairs(k,:);
    if isequal(b, [0 0]),     symbols(k) = 1+1i;
    elseif isequal(b, [0 1]), symbols(k) = -1+1i;
    elseif isequal(b, [1 1]), symbols(k) = -1-1i;
    else,                     symbols(k) = 1-1i;
    end
end

symbols = symbols / sqrt(2);

figure;
scatter(real(symbols), imag(symbols), 'filled');
grid on; axis([-1.5 1.5 -1.5 1.5]);
xlabel('In-Phase'); ylabel('Quadrature');
title('Transmitted QPSK Constellation');

Nsym = Nfft;
numBlocks = floor(length(symbols)/Nsym);
symbols = symbols(1:numBlocks*Nsym);
dataBlocks = reshape(symbols, Nsym, numBlocks);
ofdmTime = ifft(dataBlocks, Nfft);

figure;
plot(real(ofdmTime(:,1)), 'o-');
grid on;
xlabel('Sample index'); ylabel('Amplitude');
title('Time-domain OFDM symbol (real part) - Block 1');

cpBlocks = [ofdmTime(end-Ncp+1:end, :); ofdmTime];
txSignal = cpBlocks(:).';

% PAPR ANALYSIS (channel-independent)
instPower = abs(ofdmTime).^2;
peakPower = max(instPower, [], 1);
avgPower  = mean(instPower, 1);
paprPerBlock_dB = 10*log10(peakPower ./ avgPower);

figure;
[f, x] = ecdf(paprPerBlock_dB);
plot(x, 1-f, 'LineWidth', 1.5);
grid on; set(gca,'YScale','log');
xlabel('PAPR_0 (dB)'); ylabel('CCDF: P(PAPR > PAPR_0)');
title('PAPR CCDF of OFDM Signal');

fprintf('Mean PAPR: %.2f dB, Max PAPR: %.2f dB\n', mean(paprPerBlock_dB), max(paprPerBlock_dB));

refSignalPower = mean(abs(txSignal).^2);   % FIXED reference power (unfaded signal)

% RUN BOTH CHANNELS AUTOMATICALLY 
fprintf('\nRunning AWGN sweep...\n');
berAWGN = runBerSweep('AWGN', txSignal, bits, EbN0_dB, bitsPerSymbol, Nfft, Ncp, numBlocks, numTrials, refSignalPower);

fprintf('Running Rayleigh sweep...\n');
berRayleigh = runBerSweep('Rayleigh', txSignal, bits, EbN0_dB, bitsPerSymbol, Nfft, Ncp, numBlocks, numTrials, refSignalPower);

% PLOTS 
figure;
semilogy(EbN0_dB, berAWGN, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('Bit Error Rate');
title('OFDM BER Performance - AWGN Channel');

figure;
semilogy(EbN0_dB, berRayleigh, 's-', 'LineWidth', 1.5, 'Color', [0.85 0.33 0.10]);
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('Bit Error Rate');
title('OFDM BER Performance - Rayleigh Channel');

% Combined comparison plot (this is the key result figure)
figure;
semilogy(EbN0_dB, berAWGN, 'o-', 'LineWidth', 1.5); hold on;
semilogy(EbN0_dB, berRayleigh, 's-', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('Bit Error Rate');
title(sprintf('OFDM BER Performance: AWGN vs Rayleigh (QPSK, Nfft=%d, Ncp=%d)', Nfft, Ncp));
legend('AWGN', 'Rayleigh');
hold off;

% SUMMARY 
fprintf('\n--- Simulation Summary ---\n');
fprintf('Nfft = %d, Ncp = %d, Total bits = %d, Blocks = %d\n', Nfft, Ncp, nBitsTotal, numBlocks);
fprintf('Mean PAPR = %.2f dB\n', mean(paprPerBlock_dB));
disp('BER at each Eb/N0 (dB):');
disp(table(EbN0_dB', berAWGN', berRayleigh', 'VariableNames', {'EbN0_dB','BER_AWGN','BER_Rayleigh'}));


% LOCAL FUNCTION 
function berResults = runBerSweep(channelType, txSignal, bits, EbN0_dB, bitsPerSymbol, Nfft, Ncp, numBlocks, numTrials, refSignalPower)
    % Runs the full receiver pipeline across all Eb/N0 points for the given
    % channelType ('AWGN' or 'Rayleigh') and returns the BER array.
    berResults = zeros(size(EbN0_dB));

    for idx = 1:length(EbN0_dB)
        snr_dB = EbN0_dB(idx) + 10*log10(bitsPerSymbol) + 10*log10(Nfft/(Nfft+Ncp));
        snrLinear = 10^(snr_dB/10);
        noisePower = refSignalPower / snrLinear;   % FIXED noise power, computed once per Eb/N0 point

        totalErrors = 0;
        totalBits = 0;

        for trial = 1:numTrials
            if strcmp(channelType, 'AWGN')
                noise = sqrt(noisePower/2) * (randn(size(txSignal)) + 1i*randn(size(txSignal)));
                rxSignal = txSignal + noise;
                h = 1;
            elseif strcmp(channelType, 'Rayleigh')
                h = (randn + 1i*randn)/sqrt(2);
                noise = sqrt(noisePower/2) * (randn(size(txSignal)) + 1i*randn(size(txSignal)));
                rxSignal = h*txSignal + noise;   % noise power stays FIXED, not renormalized to faded signal
            end

            rxBlocks = reshape(rxSignal, Nfft+Ncp, numBlocks);
            rxNoCp   = rxBlocks(Ncp+1:end, :);
            rxFreq   = fft(rxNoCp, Nfft);

            if strcmp(channelType, 'Rayleigh')
                rxFreq = rxFreq / h;
            end

            rxSymbols = rxFreq(:).';

            rxBits = zeros(1, 2*length(rxSymbols));
            for k = 1:length(rxSymbols)
                s = rxSymbols(k);
                if real(s) >= 0 && imag(s) >= 0,      b = [0 0];
                elseif real(s) < 0 && imag(s) >= 0,   b = [0 1];
                elseif real(s) < 0 && imag(s) < 0,    b = [1 1];
                else,                                  b = [1 0];
                end
                rxBits(2*k-1:2*k) = b;
            end

            txBitsTrimmed = bits(1:length(rxBits));
            totalErrors = totalErrors + sum(rxBits ~= txBitsTrimmed);
            totalBits = totalBits + length(rxBits);
        end

        berResults(idx) = totalErrors / totalBits;
    end
end
