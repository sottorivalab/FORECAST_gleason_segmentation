#!/bin/bash

currentPath=$(dirname "${0}")

tilePath="${currentPath}/../../tiles/tiles_norm_g/"
segmentedTilePath="${currentPath}/../../tiles/masks/"
resultsPath="${currentPath}/../../results/gleason/"

gleasonSegmentationModelPath="${currentPath}/../../models/GleasonModel.h5"

classificationBatchSize=1

blockWidth=500
blockHeight=500
blockOffsetX=250
blockOffsetY=250
reducedBlockWidth=224
reducedBlockHeight=224

blockLevels="[0, 1]"
labelColours="[[1.0, 1.0, 1.0], [0.0, 0.619607843, 0.450980392], [0.0, 0.619607843, 0.450980392], [0.941176471, 0.894117647, 0.258823529], [0.835294118, 0.368627451, 0.0], [0.364705882, 0.22745098, 0.607843137]]"

gleasonCodePath="${currentPath}/gleason_segmentation/"
tifCodePath="${currentPath}/tif/"

source activate pytorch0p4

if [ $# -gt 0 ]; then
    all_files=("$tilePath"/*/)
    files=()
    for var in "$@"; do
        files+=("${all_files[$((var-1))]}")
    done
else
    files=("$tilePath"/*/)
fi

for file in "${files[@]}"; do
    imageName="$(basename "$file")"
    gleasonProbabilityTilePath="${resultsPath}/Probabilities/"
    gleasonLabelTilePath="${resultsPath}/AnnotatedTiles/"
    tifPath="${resultsPath}/tif/${imageName%.*}_Mask.tif"

    imageWidth=$(sed -n 's/iWidth=//p' "${tilePath}/${imageName}/FinalScan.ini" | head -1)
    imageHeight=$(sed -n 's/iHeight=//p' "${tilePath}/${imageName}/FinalScan.ini" | head -1)
    tileWidth=$(sed -n 's/iImageWidth=//p' "${tilePath}/${imageName}/FinalScan.ini")
    tileHeight=$(sed -n 's/iImageHeight=//p' "${tilePath}/${imageName}/FinalScan.ini")

    python3 -c "import sys; sys.path.append('${gleasonCodePath}'); from ResNetUNet import ResNetUNet; from ResNetUNetEnsemble import ResNetUNetEnsemble; import segmentTiles_MultiLevel; segmentTiles_MultiLevel.segmentTiles_MultiLevel('${tilePath}/${imageName}/', '${gleasonSegmentationModelPath}', '${gleasonProbabilityTilePath}/${imageName}/',  '${gleasonLabelTilePath}/${imageName}/', [${imageWidth}, ${imageHeight}], segmentation_path='${segmentedTilePath}/${imageName}/', tile_size=[${tileWidth}, ${tileHeight}], block_size=[${blockWidth}, ${blockHeight}], block_offset=[${blockOffsetX}, ${blockOffsetY}], block_in_size=[${reducedBlockWidth}, ${reducedBlockHeight}], block_levels=${blockLevels}, label_colours=${labelColours}, ignore_border=0, batch_size=${classificationBatchSize})"

    matlab -nodesktop -nosplash -r "addpath(genpath('${tifCodePath}')); Tiles2TIF('${gleasonLabelTilePath}/${imageName}/', [${tileWidth} ${tileWidth}], [${imageWidth}, ${imageHeight}], '${tifPath}', 'png', false, false); exit;"
done
