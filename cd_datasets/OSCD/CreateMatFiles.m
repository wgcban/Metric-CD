% This script will generate mat files of rgb images of OSCD dataset
clear all;
close all;
clc;

ImageNames = ["aguasclaras","bercy","bordeaux","nantes","paris","rennes","saclay_e","abudhabi","cupertino","pisa","beihai","hongkong","beirut","mumbai","brasilia","montpellier","norcia","rio","saclay_w","valencia","dubai","lasvegas","milano","chongqing"];

for i = 1:1:length(ImageNames)
    ImageName = ImageNames(i);
    ImageDir = strcat("./Onera Satellite Change Detection dataset - Images/", ImageName, "/pair/");
    Img1 = imread(ImageDir+"img1.png");
    Img2 = imread(ImageDir+"img2.png");
    
    [h, w, ~] = size(Img1);
    
    w = min(w, h);
    h = min(w, h);
    
    preChangeImage = double(imresize(Img1, [w, h]));
    postChangeImage = double(imresize(Img2, [w, h]));
    
    
    LabelDir = strcat("./Onera Satellite Change Detection dataset - Labels/",ImageName,"/cm/", "cm.png");
    Label = imread(LabelDir);
    if size(Label,3)==3
        Label = double(Label(:,:,1));
    end
    Label = double(imresize(Label, [w, h], 'nearest'))./255;
    save("./ImagesMat/"+ImageName+".mat", 'preChangeImage', 'postChangeImage', 'Label');
end