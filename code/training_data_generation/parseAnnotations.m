function Annotations = parseAnnotations(AnnotationPath, TilePath, ShowInactive)
    if nargin < 3
        ShowInactive = false;
    end
    
    annotationText = fileread(AnnotationPath);
    annotationText = regexprep(annotationText, '[\[()\]]', '');
    lines = textscan(annotationText, '%s', 'Delimiter', '\n');
    
    annos = cellfun(@(x) split(x, ','), lines{1}, 'UniformOutput', false);
    annotationActive = cellfun(@(x) str2double(x(1)), annos);
    
    fScanText = fileread(fullfile(TilePath,  'FinalScan.ini'));
    iWidth = regexp(fScanText, '(iWidth=)(\d*)', 'tokens');
    iHeight = regexp(fScanText, '(iHeight=)(\d*)', 'tokens');
    imageSize = [str2double(iWidth{1}{2}), str2double(iHeight{1}{2})];
    
    if ~ShowInactive
        annos = annos(annotationActive==1);
    end
    
    annotationTypes = cellfun(@(x) str2double(x(3)), annos);
    
    ClosedFreehandAnnos = {};
    ArrowAnnos = {};
    LineAnnos = {};
    RectangleAnnos = {};
    CircleAnnos = {};
    PointAnnos = {};
    TextAnnos = {};
    
    for i=1:length(annos)
        switch(annotationTypes(i))
            case 0
                x1 = str2double(annos{i}{4});
                y1 = str2double(annos{i}{5});
                x2 = str2double(annos{i}{6});
                y2 = str2double(annos{i}{7});
                
                colour = annos{i}{end-2};
                
                LineAnnos = [LineAnnos; {'', [x1 y1; x2 y2].*imageSize(1), colour}];
            case 1
                x1 = str2double(annos{i}{4});
                y1 = str2double(annos{i}{5});
                x2 = str2double(annos{i}{6});
                y2 = str2double(annos{i}{7});
                
                colour = annos{i}{end-2};
                
                ArrowAnnos = [ArrowAnnos; {'', [x1 y1; x2 y2].*imageSize(1), colour}];
            case 2
                x1 = str2double(annos{i}{4});
                y1 = str2double(annos{i}{5});
                x2 = str2double(annos{i}{6});
                y2 = str2double(annos{i}{7});
                
                colour = annos{i}{end-2};
                
                RectangleAnnos = [RectangleAnnos; {'', [x1 y1; x2 y2].*imageSize(1), colour}];
            case 3
                x1 = str2double(annos{i}{4});
                y1 = str2double(annos{i}{5});
                x2 = str2double(annos{i}{6});
                y2 = str2double(annos{i}{7});
                
                centre = [(x1+x2) (y1+y2)]/2;
                axisX = x2-x1;
                axisY = y2-y1;
                
                angles = linspace(0, 2*pi, 10000)';
                points = ([axisX*sin(angles) axisY*cos(angles)]+centre).*imageSize(1);
                colour = annos{i}{end-2};
                
                CircleAnnos = [CircleAnnos; {'', points, colour}];
            case 4
                points = reshape(cellfun(@(x) str2double(x).*imageSize(1), annos{i}(4:end-3)), 2, [])';
                colour = annos{i}{end-2};
                
                ClosedFreehandAnnos = [ClosedFreehandAnnos; {'', points, colour}];
            case 5
                x = str2double(annos{i}{4});
                y = str2double(annos{i}{5});
                text = annos{i}{6}(2:end-1);
                colour = annos{i}{end-2};
                
                TextAnnos = [TextAnnos; {text, [x y].*imageSize(1), colour}];
            case 6
                x = str2double(annos{i}{4});
                y = str2double(annos{i}{5});
                colour = annos{i}{end-2};
                
                PointAnnos = [PointAnnos; {'', [x y].*imageSize(1), colour}];
        end
    end
    
    Annotations = struct;
    Annotations.ClosedFreehands = ClosedFreehandAnnos;
    Annotations.Points = PointAnnos;
    Annotations.Circles = CircleAnnos;
    Annotations.Lines = LineAnnos;
    Annotations.Arrows = ArrowAnnos;
    Annotations.Rectangles = RectangleAnnos;
    Annotations.Text = TextAnnos;
end

