require 'torch'
require 'nn'
require 'cunn' 
require 'cudnn'
require 'paths'
require 'cutorch'
local cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'
model = torch.load('/home/gss/Code/GANs_code/Action_Prediction/model_10000.bin')
test = torch.load("/home/gss/Code/GANs_code/Action_Prediction/test_full_testlist01_label.t7")
testSamples = #test
idx = torch.randperm(testSamples)  --Shuffling Data

--Loading Mean Data
fmeans = torch.load('test_fmeans.dat')

local test_input = torch.CudaTensor(1, 3, 16, 112, 112)   -- partial video description

num = 0
for i = 1, testSamples do
  print('Line: ', i, '/ ', testSamples)
  linenumber = idx[i]
  gt_label = test[linenumber][2]
  file_name = '/home/gss/Code/GANs_code/Action_Prediction/' .. test[linenumber][1]
  cap  = cv.VideoCapture{file_name}
  print(test[linenumber][1])
  nF = cap:get{cv.CAP_PROP_FRAME_COUNT}  
  print(nF)
  --Looping Frames
  for f=0,nF-1 do  
    cap:set{1,f}
    ret,frame = cap:read{}       
    frame = frame:permute(3,1,2)  --(Length, Width, Height)     
    --Looping Channels (For C3D Preparation)
    for c=1,3 do      
      test_input[1][c][f+1] = (frame[c] - fmeans[c])--:div(255.0)  #
    end --End of Channels Loop    
  end
  cap:release()
  predict = model:forward(test_input):clone()
  _,pre_label = torch.max(predict,1)
  print('predict:', pre_label[1])
  print('groundtruth', gt_label)
  if (pre_label[1] == gt_label) then  num = num + 1 end
end
print('accuracy is :')
print(num / testSamples)