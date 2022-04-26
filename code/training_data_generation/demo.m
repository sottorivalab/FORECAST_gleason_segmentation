[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));

ImageTilePath = fullfile(scriptDir, '..', '..', 'tiles', 'tiles_norm_g');
AnnotationPath = fullfile(scriptDir, '..', '..', 'training_data', 'annotation_files');
PatchPath = fullfile(scriptDir, '..', '..', 'training_data', 'new_patches');

Levels = [0 1];
MaskLevel = 0;
% InPatchSize = [500 500];
OutPatchSize = [500 500];

Labels = {'Normal', 'PIN', 'Gleason3', 'Gleason4', 'Gleason5', 'Neg'};
SkipLabels = {'Neg'};
Colours = {'#ffffff', '#800080', '#ff0000', '#ffff00', '#c0c0c0', '#0000ff'}; 
nAnnos = 10;

colourLookup = containers.Map(Colours, Labels);

ImageTileDirs = dir(ImageTilePath);

ImageTileDirs = ImageTileDirs(~ismember({ImageTileDirs.name}, {'.', '..'}));
ImageTileDirs = cellfun(@(x, y) fullfile(x, y), {ImageTileDirs.folder}', {ImageTileDirs.name}', 'UniformOutput', false);

[~, fNames, ~] = cellfun(@fileparts, ImageTileDirs, 'UniformOutput', false);
AnnotationFiles = cellfun(@(x) fullfile(AnnotationPath, [x '.txt']), fNames, 'UniformOutput', false);

Annotations = cellfun(@(x, y) parseAnnotations(x, y), AnnotationFiles, ImageTileDirs, 'UniformOutput', false);
AnnotationLabels = cellfun(@(x) cellfun(@(y) colourLookup(y), x.ClosedFreehands(ismember(x.ClosedFreehands(:, 3), Colours), 3), 'UniformOutput', false), Annotations, 'UniformOutput', false);
Annotations = cellfun(@(x) x.ClosedFreehands(ismember(x.ClosedFreehands(:, 3), Colours), 2), Annotations, 'UniformOutput', false);

for i=1:length(ImageTileDirs)
    [~, fName, ~] = fileparts(ImageTileDirs{i});
    MaskPath = fullfile(PatchPath, fName);
    
    if ~isfolder(MaskPath)
        mkdir(MaskPath);
    end
    
    PatchParams = GenerateMultiLevelPatches(ImageTileDirs{i}, Annotations{i}, nAnnos, MaskPath, MaskLevel, Levels, 0, InPatchSize, OutPatchSize, 'Mask');

    for j=1:length(Labels)
        if ~ismember(Labels{j}, SkipLabels)
            GenerateMultiLevelPatches(ImageTileDirs{i}, Annotations{i}(strcmp(AnnotationLabels{i}, Labels{j})), PatchParams, MaskPath, MaskLevel, Levels, 0, InPatchSize, OutPatchSize, Labels{j});
        end
    end
end
