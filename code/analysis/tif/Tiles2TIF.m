function Tiles2TIF( TilePath, TileSize, ImageSize, OutFile, Ext, Overwrite, JPEGCompression)
%Tiles2TIF Summary of this function goes here
%   Detailed explanation goes here
    
    if nargin < 5
        Ext = 'jpg';
    end

    if nargin < 6
        Overwrite = true;
    end

    if nargin < 7
        JPEGCompression = true;
    end
    
    if ischar(TileSize)
        TileSize = str2num(TileSize);
    end
    
    if ischar(ImageSize)
        ImageSize = str2num(ImageSize);
    end

    if Overwrite || ~isfile(OutFile)
        [OutFolder, ~, ~] = fileparts(OutFile);
        mkdir(OutFolder);

        t = Tiff(OutFile, 'w8');
        setTag(t,'ImageWidth',double(ImageSize(1)));
        setTag(t,'ImageLength',double(ImageSize(2)));
        setTag(t,'Photometric',Tiff.Photometric.RGB);
        if JPEGCompression
            setTag(t,'Compression',Tiff.Compression.JPEG);
            setTag(t,'JPEGQuality',95);
        else
            setTag(t,'Compression',Tiff.Compression.LZW);
        end
        setTag(t,'BitsPerSample',8);
        setTag(t,'SamplesPerPixel',3);
        setTag(t,'TileWidth',double(TileSize(1)));
        setTag(t,'TileLength',double(TileSize(2)));
        setTag(t,'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
        setTag(t,'XResolution', 0.22);
        setTag(t,'YResolution', 0.22);
        
        tileGrid = ceil(ImageSize(:)./TileSize(:));

        nTiles = length(dir(fullfile(TilePath, ['Da*.' Ext])));

        if nTiles == 0
           t.close();
           delete(OutFile);
           error(['No tiles found in ' TilePath '!']);
        else
            for i=1:prod(tileGrid)
                fprintf('Writing tile %d of %d to tif\n', i, nTiles);

                tileFile = fullfile(TilePath, ['Da' num2str(i-1) '.' Ext]);

                if isfile(tileFile)
                    im = imread(tileFile);
        
                    if ndims(im) || size(im, 3) == 1
                        im = repmat(im, [1 1 3]);
                    end

                    tilePosition = [mod(i-1, tileGrid(1)), floor((i-1)./tileGrid(1))];

                    expectedTileSize = min(TileSize(:), ImageSize(:)-(TileSize(:).*tilePosition(:)));

                    if size(im, 1) ~= expectedTileSize(2) || size(im, 2) ~= expectedTileSize(1)
                        im = imresize(im, expectedTileSize(2:-1:1));
                    end
            
                    writeEncodedTile(t, i, im);
                else
                    t.close();
                    delete(OutFile);
                    error(['Tile Da' num2str(i-1) '.' Ext ' not found!']);
                end
            end
        end

        t.close();
    else
        fprintf('TIF file already exists, skipping.');
    end
end

