%% create_spectograms:
% this function creates the spectrogram for each of the electrodes under each of the conditions.
% Additionally, displays spectrograms showing the difference between right and
% left inputs in each of the electrodes.

% inputs:
% data_cell - A cell array, contains the data saperating by cnditions and channels.
% window - The size of the window for the FFT process.
% noverlap -  The size of the overlap for the FFT process.
% SR - Sampling rate.
% Titles,FontSize - structs, contains the titles end the font size.

% output:
% 2 figures:
% 1- contains Spectrogram for Each Condition By Each Channel.
% 2- contains the Difference Spectrum of the conditions


function create_spectograms(data_cell, window, noverlap, SR, Titles,FontSize)
%%
fig1 = figure;
n=1;
for ichannel= 1:size(data_cell,1)
    for iLR= 1:size(data_cell,1)
        for j=1:size(data_cell{ichannel, iLR},1)
            [~, w, t, ps{ichannel, iLR}(:, :, j)] = spectrogram(data_cell{ichannel, iLR}(j, :), window, noverlap, [], SR, 'power');
        end
        spec{ichannel, iLR} = mean(10*log10(ps{ichannel, iLR}), 3); % convert to dB
        subplot(2, 2, n)
        imagesc(t, w, spec{ichannel, iLR});
        n=n+1;
        sgtitle('Spectrogram for Each Condition By Each Channel', 'FontSize', FontSize.L);
        title([Titles.RnL{iLR}, ' - ', Titles.Channels{ichannel}], 'fontSize', FontSize.L);
        axis xy % Flip axis
        cb = colorbar ; % Add colorbar
        cb.Label.String = 'Power [dB]';
        cb.Label.FontSize = FontSize.M;
        C = axes(fig1, 'visible', 'off');
        C.XLabel.Visible = 'on';
        C.YLabel.Visible = 'on';
        xlabel('Time [sec]', 'FontSize', FontSize.L, 'FontWeight', 'bold');
        ylabel('Frequency [Hz]', 'FontSize', FontSize.L, 'FontWeight', 'bold');
        set(gca,'FontSize',FontSize.L)
        
    end
end
%%
fig2=figure;
for iLR=1:size(spec,2)
    diff{iLR,1}=spec{iLR,1}-spec{iLR,2};
    subplot(1,size(spec,2),iLR)
    imagesc(t, w, diff{iLR,1});
    axis xy % Flip axis
    title(Titles.Channels(iLR), 'FontSize', FontSize.L)
    sgtitle(Titles.diff, 'FontSize', FontSize.L,'FontWeight', 'bold')
    cb = colorbar ;
    cb.Label.String = 'Power [dB]';
    cb.Label.FontSize = FontSize.M;
    C = axes(fig2, 'visible', 'off');
    C.XLabel.Visible = 'on';
    C.YLabel.Visible = 'on';
    xlabel('Time [sec]', 'FontSize', FontSize.L, 'FontWeight', 'bold');
    ylabel('Frequency [Hz]', 'FontSize', FontSize.L, 'FontWeight', 'bold');
    set(gca,'FontSize',FontSize.L)
    
end
end
