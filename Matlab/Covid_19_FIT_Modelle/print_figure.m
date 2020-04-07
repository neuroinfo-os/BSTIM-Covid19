function print_figure(formats,figure_Name)


% Example
% formats.Fig =1;
% formats.JPG =1;
% formats.Res =150;
% formats.PDF =1;
% formats.eps =1;
if isfield(formats,'Fig')==1
    if formats.Fig ==1
        saveas(gca,figure_Name)
    end
end
if isfield(formats,'JPG')==1
    if formats.JPG ==1
        Printcommand=['print -djpeg -r' num2str(formats.Res)  ' ' figure_Name ,         '.jpg' ];
        eval(Printcommand);
    end
end
if isfield(formats,'PDF')==1
    if formats.PDF ==1
        Printcommand=['print -dpdf ',   figure_Name ,         '.pdf' ];
        eval(Printcommand);
    end
end
if isfield(formats,'eps')==1
    if formats.eps ==1
        Printcommand=['print -depsc ',   figure_Name ,         '.eps' ];
        eval(Printcommand);
    end
end