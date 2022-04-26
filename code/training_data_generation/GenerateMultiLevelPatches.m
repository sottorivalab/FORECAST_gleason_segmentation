function PatchParams = GenerateMultiLevelPatches(TilePath, Boundaries, NPatches, PatchPath, MaskLevel, ImageLevels, Sampling, InPatchSize, OutPatchSize, MaskName)
    if nargin < 10
        MaskName = 'Mask';
    end

    if nargin < 9
        OutPatchSize = [2000 2000];
    end

    if nargin < 8
        InPatchSize = [2000 2000];
    end

    if nargin < 7
        Sampling = 0;
    end

    if nargin < 6
        ImageLevels = (1:3);
    else
        ImageLevels = sort(ImageLevels);
    end
    
    if nargin < 5
        MaskLevel = 1;
    end
    
    if size(NPatches, 2) > 1
        PatchParams = NPatches;
        NPatches = size(PatchParams, 1);
    else
        PatchParams = nan(NPatches, 3);
    end
    
    minProp = 0;

    [~, imName, ~] = fileparts(TilePath);
    
    fScanText = fileread(fullfile(TilePath,  'FinalScan.ini'));
    tWidth = regexp(fScanText, '(iImageWidth=)(\d*)', 'tokens');
    tHeight = regexp(fScanText, '(iImageHeight=)(\d*)', 'tokens');
    TileSize = [str2double(tWidth{1}{2}), str2double(tHeight{1}{2})];
    iWidth = regexp(fScanText, '(iWidth=)(\d*)', 'tokens');
    iHeight = regexp(fScanText, '(iHeight=)(\d*)', 'tokens');
    ImageSize = [str2double(iWidth{1}{2}), str2double(iHeight{1}{2})];
   
    annotationSelectionMask = false(ceil(ImageSize./(InPatchSize.*(2.^MaskLevel))));
    
    minPixels = minProp*InPatchSize(1)*InPatchSize(2);
    
    if Sampling == 0 || Sampling == 1
        for i=1:size(annotationSelectionMask, 1)
            for j=1:size(annotationSelectionMask, 2)
                test1 = any(cellfun(@(x) any(x(:, 1) >= (i-1).*InPatchSize(1).*(2.^MaskLevel) & x(:, 1) < i.*InPatchSize(1).*(2.^MaskLevel) & x(:, 2) >= (j-1).*InPatchSize(2).*(2.^MaskLevel) & x(:, 2) < j.*InPatchSize(2).*(2.^MaskLevel)), Boundaries));
                test2 = any(cellfun(@(x) WindingCheck((i-1).*InPatchSize.*(2.^MaskLevel), x) ~= 0, Boundaries));
                annotationSelectionMask(i, j) = test1 | test2;
            end
        end
    end
    
    if Sampling == 0
        [goodX, goodY] = find(annotationSelectionMask);
    elseif Sampling == 1
        [goodX, goodY] = find(~annotationSelectionMask);
    elseif Sampling == 2
        [goodX, goodY] = meshgrid(1:size(annotationSelectionMask, 1), 1:size(annotationSelectionMask, 2));
    else
        error(['Mode ' num2str(Sampling) ' is not recognised.']);
    end
    
    Boundaries = cellfun(@(x) pathLengthParameterisation(x, 'pathLength', 0.5), Boundaries, 'UniformOutput', false);
    
    mkdir(fullfile(PatchPath, MaskName));
    
    for i=1:length(ImageLevels)
        mkdir(fullfile(PatchPath, ['Level' num2str(ImageLevels(i))]));
    end
    
    generatedPatches = 0;
    
    while generatedPatches < NPatches
        if isnan(PatchParams(generatedPatches+1, 1))
            PatchBlockIdx = randi(length(goodX));
            imageCentre = floor(([goodX(PatchBlockIdx) goodY(PatchBlockIdx)]-0.5+rand(1, 2)).*InPatchSize.*(2.^MaskLevel));
            angle = rand()*2*pi;
        else
            imageCentre = PatchParams(generatedPatches+1, 1:2);
            angle = PatchParams(generatedPatches+1, 3);
        end
        
        imageRegion = [imageCentre-floor((InPatchSize.*(2.^MaskLevel)-1)./2) imageCentre+ceil((InPatchSize.*(2.^MaskLevel)-1)./2)];
        tMat = [1 0 0; 0 1 0; -imageCentre 1]*[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0; 0 0 1]*[1 0 0; 0 1 0; imageCentre 1];
        
        mask = false(InPatchSize);
        
        for i=1:length(Boundaries)
            tBoundary = [Boundaries{i} ones(size(Boundaries{i}, 1), 1)]/tMat;
            tBoundary = tBoundary(:, 1:2)./tBoundary(:, 3);
            tBoundary = floor((tBoundary - imageRegion([1 2])).*(2.^-MaskLevel));
            
            tBoundaryDiff = circshift(tBoundary, 1, 1)-tBoundary;
            dups = tBoundaryDiff(:, 1) == 0 & tBoundaryDiff(:, 2) == 0;
            tBoundary(dups, :) = [];
            tBoundaryDiff(dups, :) = [];
            
            inMaskWindow = (tBoundary(:, 1)>=0 & tBoundary(:, 2)>=0 & tBoundary(:, 1)<size(mask, 2) & tBoundary(:, 2)<size(mask, 1));
            
            if all(inMaskWindow)
                subMask = zeros(size(mask));
                subMask(sub2ind(size(subMask), tBoundary(:, 2)+1, tBoundary(:, 1)+1)) = true;
                subMask = imfill(subMask, 'holes');
                
                mask = mask | subMask;
            elseif ~any(inMaskWindow)
                    x = size(mask, 2)/2;
                    y = size(mask, 1)/2;
                    
                    x0 = 2*abs(max(tBoundary(:, 1)));
                    y0 = y-2;

                    xDiff = x-1-x0;
                    yDiff = y-1-y0;

                    tN = xDiff.*(tBoundary(:, 2)-y0) - yDiff.*(tBoundary(:, 1)-x0);
                    sN = tBoundaryDiff(:, 1).*(tBoundary(:, 2)-y0) - tBoundaryDiff(:, 2).*(tBoundary(:, 1)-x0);
                    D = tBoundaryDiff(:, 1).*yDiff - tBoundaryDiff(:, 2).*xDiff;

                    tN(D<0) = -tN(D<0);
                    sN(D<0) = -sN(D<0);
                    D(D<0) = -D(D<0);
                    
                    intersections = nnz(tN >= 0 & tN <= D & sN >= 0 & sN <= D);
                    
                if mod(intersections, 2) ~= 0
                    mask(:, :) = true;
                    break;
                end
            else
                subMask = false(size(mask));
                subMask(sub2ind(size(subMask), tBoundary(inMaskWindow, 2)+1, tBoundary(inMaskWindow, 1)+1)) = true;
                cc = bwconncomp(~subMask, 4);
                    
                for j=1:cc.NumObjects
                    [y, x] = ind2sub(size(subMask), cc.PixelIdxList{j}(1));
                    
                    x0 = 2*abs(max(tBoundary(:, 1)));
                    y0 = y-2;

                    xDiff = x-1-x0;
                    yDiff = y-1-y0;

                    tN = xDiff.*(tBoundary(:, 2)-y0) - yDiff.*(tBoundary(:, 1)-x0);
                    sN = tBoundaryDiff(:, 1).*(tBoundary(:, 2)-y0) - tBoundaryDiff(:, 2).*(tBoundary(:, 1)-x0);
                    D = tBoundaryDiff(:, 1).*yDiff - tBoundaryDiff(:, 2).*xDiff;

                    tN(D<0) = -tN(D<0);
                    sN(D<0) = -sN(D<0);
                    D(D<0) = -D(D<0);
                    
                    intersections = nnz(tN >= 0 & tN <= D & sN >= 0 & sN <= D);
                    
                    if mod(intersections, 2) ~= 0
                        subMask = imfill(subMask, [y, x], 4);
                    end
                end
                
                mask = mask | subMask;
            end
        end
        
        if ~isnan(PatchParams(generatedPatches+1, 1)) || (Sampling == 0 && nnz(mask(:)) > minPixels) || (Sampling == 1 && nnz(~mask(:)) > minPixels) || (Sampling == 2)
            imwrite(imresize(mask, OutPatchSize), fullfile(PatchPath, MaskName, [imName '_' num2str(imageCentre(1)) '_' num2str(imageCentre(2)) '_' num2str(round(180*angle/pi)) '.png']));
            
            for i=1:length(ImageLevels)
                levelPath = fullfile(PatchPath, ['Level' num2str(ImageLevels(i))], [imName '_' num2str(imageCentre(1)) '_' num2str(imageCentre(2)) '_' num2str(round(180*angle/pi)) '.png']);
                
                if ~isfile(levelPath)
                    imageRegion = [imageCentre-floor((InPatchSize.*(2^ImageLevels(i))-1)./2) imageCentre+ceil((InPatchSize.*(2^ImageLevels(i))-1)./2)];
                    imageRegion = imageRegion([1 2; 1 4; 3 2; 3 4]);

                    tImageRegion = [imageRegion ones(length(imageRegion), 1)]*tMat;
                    tImageRegion = tImageRegion(:, 1:2)./tImageRegion(:, 3);
                    tImageRegion = [min(tImageRegion, [], 1) max(tImageRegion, [], 1)];

                    loadRegion = tImageRegion.*(2.^-ImageLevels(i)) + 1;
                    loadRegion = [floor(loadRegion(1:2)) ceil(loadRegion(3:4))];

                    padding = [max(1-loadRegion(1:2), 0) max(loadRegion(3:4)-floor(ImageSize*(2.^-ImageLevels(i))), 0)];
                    loadRegion = [loadRegion(1:2)+padding(1:2), loadRegion(3:4)-padding(3:4)];
                    
                    image = imread_cws(TilePath, ImageSize, TileSize, ImageLevels(i), {loadRegion([2 4]), loadRegion([1 3])});
                    image = padarray(image, padding([2, 1]), 255, 'pre');
                    image = padarray(image, padding([4, 3]), 255, 'post');
                    
                    worldCoordRef = imref2d(size(image), tImageRegion([1 3]), tImageRegion([2 4]));
                    localCoordRef = imref2d(InPatchSize, imageRegion([1 4], 1)', imageRegion([1 4], 2)');

                    tImage = imwarp(image, worldCoordRef, invert(affine2d(tMat)), 'OutputView', localCoordRef, 'FillValues', 255);

                    imwrite(imresize(tImage, OutPatchSize), levelPath);
                end
            end
            
            if isnan(PatchParams(generatedPatches+1, 1))
                PatchParams(generatedPatches+1, :) = [imageCentre angle];
            end
            
            generatedPatches = generatedPatches+1;
        end
    end
end

function I = imread_cws(TilePath, ImageSize, TileSize, ReductionLevel, PixelRegion, ext)
    if nargin < 6
        ext = '.jpg';
    end
    
    I = zeros([PixelRegion{1}(2)-PixelRegion{1}(1)+1, PixelRegion{2}(2)-PixelRegion{2}(1)+1, 3]);
    
    TileGrid = ceil(ImageSize./TileSize);
    
    ScaledPixelRegion = {(PixelRegion{1}-1).*(2.^ReductionLevel) + 1, (PixelRegion{2}-1).*(2.^ReductionLevel) + 1};
    
    XTiles = floor(((ScaledPixelRegion{2}(1)-1)./TileSize(1))):floor(((ScaledPixelRegion{2}(2)-1)./TileSize(1)));
    YTiles = floor(((ScaledPixelRegion{1}(1)-1)./TileSize(2))):floor(((ScaledPixelRegion{1}(2)-1)./TileSize(2)));
    
    XPositionsIn = min(max(XTiles'*TileSize(1) + [1 TileSize(1)], ScaledPixelRegion{2}(1)), ScaledPixelRegion{2}(2));
    YPositionsIn = min(max(YTiles'*TileSize(2) + [1 TileSize(2)], ScaledPixelRegion{1}(1)), ScaledPixelRegion{1}(2));
    
    XPositionsOut = (XPositionsIn-ScaledPixelRegion{2}(1)).*(2^-ReductionLevel);
    YPositionsOut = (YPositionsIn-ScaledPixelRegion{1}(1)).*(2^-ReductionLevel);
    
    XPositionsOut = [ceil(XPositionsOut(:, 1)) floor(XPositionsOut(:, 2))]+1;
    YPositionsOut = [ceil(YPositionsOut(:, 1)) floor(YPositionsOut(:, 2))]+1;
    
    XPositionsIn = mod(XPositionsIn-1, TileSize(1))+1;
    YPositionsIn = mod(YPositionsIn-1, TileSize(2))+1;
    
    for i=1:length(XTiles)
        for j=1:length(YTiles)
            tileIdx = XTiles(i) + YTiles(j)*TileGrid(1);
            tile = imread(fullfile(TilePath, ['Da' num2str(tileIdx) ext]));
            tile = im2double(tile);
            tile = tile(YPositionsIn(j, 1):YPositionsIn(j, 2), XPositionsIn(i, 1):XPositionsIn(i, 2), :);
            
            outSize = [YPositionsOut(j, 2)-YPositionsOut(j, 1), XPositionsOut(i, 2)-XPositionsOut(i, 1)]+1;
            I(YPositionsOut(j, 1):YPositionsOut(j, 2), XPositionsOut(i, 1):XPositionsOut(i, 2), :) = imresize(tile, outSize);
        end
    end
end

function winding = WindingCheck(Point, Boundary)
    winding = 0;

    for i=1:size(Boundary, 1)
        e1 = Boundary(i, :);
        e2 = Boundary(mod(i,size(Boundary, 1))+1, :);
        
        if (e2(:, 2) > Point(:, 2) && e1(:, 2) <= Point(:, 2))
            if ((e2(:, 1) - e1(:, 1)) * (Point(:, 2) - e1(:, 2)) - (Point(:, 1) - e1(:, 1)) * (e2(:, 2) - e1(:, 2)) > 0)
                 winding = winding+1;
            end
        elseif (e2(:, 2) <= Point(:, 2) && e1(:, 2) > Point(:, 2))
            if ((e2(:, 1) - e1(:, 1)) * (Point(:, 2) - e1(:, 2)) - (Point(:, 1) - e1(:, 1)) * (e2(:, 2) - e1(:, 2)) < 0)
                 winding = winding-1;
            end
        end
    end
end



