function [paramPoints, perimeter] = pathLengthParameterisation( pointSet, in1, in2 )
%PATHLENGTHPARAMETERISATION Parametrises the boundary into a number of evenly spaced points
%   Input:
%       pointSet: A set of points defining the boundary
%
%   Output:
%       paramPoints: A set of evenly spaced points defining the boundary
%       perimeter: The length of the boundary

    pointDiff = pointSet-circshift(pointSet, -1);
    distances = sqrt(power(pointDiff(:, 1), 2) + power(pointDiff(:, 2), 2));
    
    pointSet = pointSet(distances~=0, :);
    pointDiff = pointDiff(distances~=0, :);
    distances = distances(distances~=0);
    
    perimeter = sum(distances);
    
    nPointFlag = true;
    numPoints = 2000;
    pathLength = 1;
    
    if nargin < 3
        if ischar(in1)
            if strcmp(in1, 'pathLength')
                nPointFlag = false;
            end
        else
            numPoints = in1;
        end
    else
        if ischar(in1)
            if strcmp(in1, 'pathLength')
                nPointFlag = false;
                pathLength = in2;
            elseif strcmp(in1, 'nPoints')
                numPoints = in2;
            end
        else
            numPoints = in1;
        end
    end
    
    if nPointFlag
        pathLength = perimeter/numPoints;
    else
        numPoints = floor(perimeter/pathLength);
    end
    
    paramPoints = zeros(numPoints, 2);
        
    totalDistance = 0;
    currentPoint = 1;
    
    for i=1:size(paramPoints, 1)        
        while totalDistance < pathLength*(i-1)
            totalDistance = totalDistance + distances(currentPoint);
            currentPoint = mod(currentPoint, numel(distances))+1;
        end
        
        previousPoint = mod(currentPoint - 2, numel(distances))+1;
        backDistance = distances(previousPoint);
        
        contribution = (totalDistance - (pathLength*(i-1)))/backDistance;
        
        paramPoints(i, :) = contribution*pointSet(previousPoint, :) + (1-contribution)*pointSet(currentPoint, :);
    end
end

