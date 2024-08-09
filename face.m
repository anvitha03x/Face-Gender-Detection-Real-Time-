clear;
clc;

% Initialize webcam
cam = webcam();
cam.Resolution = '1280x720';

% Set up face detector
face_Detector = vision.CascadeObjectDetector();

% Set up video player
video_Player = vision.VideoPlayer('Position', [100, 100, 1280, 720]);

% Set up point tracker
point_Tracker = vision.PointTracker('MaxBidirectionalError', 2);
run_loop = true;
number_of_Points = 0;
frame_Count = 0;

% Define paths for gender images
genderDataDir = 'C:/Users/HP/OneDrive/Documents/MATLAB/genderDataDir'; % Adjust this path
maleFolder = fullfile(genderDataDir, 'Male/images (1).png');
femaleFolder = fullfile(genderDataDir, 'Female/image (1).png');

% Load gender images
maleImages = imageDatastore(maleFolder, 'IncludeSubfolders', true, 'FileExtensions', {'.png', '.jpg', '.jpeg'});
femaleImages = imageDatastore(femaleFolder, 'IncludeSubfolders', true, 'FileExtensions', {'.png', '.jpg', '.jpeg'});

% Extract features from gender images
maleFeatures = extractFeaturesFromImages(maleImages);
femaleFeatures = extractFeaturesFromImages(femaleImages);

% Function to extract features from images
function features = extractFeaturesFromImages(imds)
    numImages = numel(imds.Files);
    features = cell(numImages, 1);
    for i = 1:numImages
        img = readimage(imds, i);
        resizedImg = imresize(img, [100, 100]); % Adjust size if needed
        features{i} = extractHOGFeatures(rgb2gray(resizedImg));
    end
end

% Function to classify gender based on features
function label = classifyGender(faceFeatures, maleFeatures, femaleFeatures)
    % Calculate Euclidean distance to each class
    distMale = pdist2(faceFeatures, cat(1, maleFeatures{:}));
    distFemale = pdist2(faceFeatures, cat(1, femaleFeatures{:}));

    % Determine the label based on the closest class
    if mean(distMale) < mean(distFemale)
        label = 'Female';
    else
        label = 'Male';
    end
end

while run_loop && frame_Count < 400
    % Capture video frame
    video_Frame = snapshot(cam);
    gray_Frame = rgb2gray(video_Frame);
    frame_Count = frame_Count + 1;

    if number_of_Points < 10
        % Detect faces
        face_Rectangle = face_Detector.step(gray_Frame);

        if ~isempty(face_Rectangle)
            disp('Face detected!'); % Debug: Confirm that a face is detected
            
            % Detect features within the face region
            points = detectMinEigenFeatures(gray_Frame, 'ROI', face_Rectangle(1, :));

            xy_Points = points.Location;
            number_of_Points = size(xy_Points, 1);
            release(point_Tracker);
            initialize(point_Tracker, xy_Points, gray_Frame);

            previous_Points = xy_Points;

            % Draw face rectangle
            rectangle = bbox2points(face_Rectangle(1, :));
            face_Polygon = reshape(rectangle', 1, []);

            % Extract face region for gender classification
            face_Region = video_Frame(face_Rectangle(1, 2):(face_Rectangle(1, 2) + face_Rectangle(1, 4) - 1), ...
                                      face_Rectangle(1, 1):(face_Rectangle(1, 1) + face_Rectangle(1, 3) - 1), :);

            % Resize face region for feature extraction
            resized_Face = imresize(face_Region, [100, 100]); % Adjust size if needed

            % Extract features from face region
            faceFeatures = extractHOGFeatures(rgb2gray(resized_Face));

            % Classify the face
            try
                genderLabel = classifyGender(faceFeatures, maleFeatures, femaleFeatures);
                disp(['Gender detected: ', genderLabel]); % Debug: Show detected gender
            catch
                genderLabel = 'Unknown'; % Default to Unknown if classification fails
                disp('Classification failed.'); % Debug: Show classification failure
            end

            % Annotate the video frame with gender label
            textPosition = [face_Rectangle(1, 1), face_Rectangle(1, 2) - 30]; % Position for the text label above the face box
            video_Frame = insertText(video_Frame, textPosition, ['Gender: ' genderLabel], 'TextColor', 'red', 'BoxOpacity', 0.4, 'FontSize', 18);

            % Draw the face rectangle and points
            video_Frame = insertShape(video_Frame, 'Polygon', face_Polygon, 'LineWidth', 3);
            video_Frame = insertMarker(video_Frame, xy_Points, '+', 'Color', 'White');
        else
            disp('No face detected.'); % Debug: No face detected
        end

    else
        % Track points
        [xy_Points, isFound] = step(point_Tracker, gray_Frame);
        new_Points = xy_Points(isFound, :);
        old_Points = previous_Points(isFound, :);

        number_of_Points = size(new_Points, 1);

        if number_of_Points >= 10
            % Estimate geometric transformation
            [xform, old_Points, new_Points] = estimateGeometricTransform(...
                old_Points, new_Points, 'similarity', 'MaxDistance', 4);

            rectangle = transformPointsForward(xform, rectangle);

            face_Polygon = reshape(rectangle', 1, []);

            video_Frame = insertShape(video_Frame, 'Polygon', face_Polygon, 'LineWidth', 3);
            video_Frame = insertMarker(video_Frame, new_Points, '+', 'Color', 'White');
            previous_Points = new_Points;
            setPoints(point_Tracker, previous_Points);
        end
    end

    % Display the video frame with annotations
    step(video_Player, video_Frame);
    run_loop = isOpen(video_Player); % Continue loop if video player is open
end

% Cleanup
clear cam;
release(video_Player);
release(point_Tracker);
release(face_Detector);
