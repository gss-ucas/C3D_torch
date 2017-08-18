--[[
  Script for Training C3D in Torch
  Mohsen Fayyaz -- Sensifai -- Vision Group
--]]

--Main Code

unpack = unpack or table.unpack
require 'nn'
require 'nnlr'
--require 'dpnn'
require 'cudnn'
require 'cunn'
require 'paths'
require 'torch'
require 'cutorch'
require 'optim'
require 'sys'
local cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'
disp = require 'display'
disp.url = 'http://localhost:8009/events'
local function main()
  cutorch.setDevice(1) --CHECK HERE!
  paths.dofile('opts-UCF101.lua')
  --paths.dofile('c3d.lua')
  paths.dofile('c3d.lua') 
  model = c3dModel 
  loadModel = false
  W = torch.load('c3d.t7')  
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  model:apply(weights_init) -- loop over all layers, applying weights_init
  convs ={1,4,7,9,12,14,17,19}
  for i,v in ipairs(convs) do     
    model.modules[1].modules[v].weight = W.modules[v].weight:clone()
    model.modules[1].modules[v].bias = W.modules[v].bias:clone()   
  end
  fc = {} --{2,5}
  for i,v in ipairs(fc) do      
    model.modules[2].modules[v].weight = W.modules[v+21].weight
    model.modules[2].modules[v].bias = W.modules[v+21].bias     
  end

  -- move everything to gpu
  model:cuda()
  config = {}    

  criterion = nn.ClassNLLCriterion()
  cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
  cudnn.verbose = false
  model = cudnn.convert(model, cudnn)
  opt.train = true
  criterion:cuda()

  -----------------------------------------------------------------------------
  -- Create model or load a pre-trained one
  if opt.modelFile then -- resume training 
    model = torch.load(opt.modelFile)
    if opt.train then
      config = torch.load(opt.configFile)
    end
  end
  
  if opt.train then
    ----------------------------------------------------------------------------- 
    --Loading Train List  
    inputData = torch.load("trainlist01.t7")
    trainSamples = #inputData
    test = torch.load("/home/gss/Code/GANs_code/Action_Prediction/test_full_testlist01_label.t7")
    testSamples = #test
    test_idx = torch.randperm(testSamples)  --Shuffling Data
    --Shuffling Data
    idx = torch.randperm(trainSamples)
  
    --Loading Mean Data
    fmeans = torch.load('fmeans.dat')
      
    params, grads = model:getParameters()
    print('Number of parameters ' .. params:nElement())
    print('Number of grads ' .. grads:nElement())

    local eta = config.eta or opt.eta 
    local momentum = config.momentum or opt.momentum
    local iter  = config.iter or 1
    local lastT  = config.lastT or 1  
    local epoch = config.epoch or 0
    local err  = 0
  
    model:training()
    --model:forget()
    baseWeightDecay = 0.05--0.0005
    eta = 0.001--0.00001
    local learningRates, weightDecays = model:getOptimConfig(eta, baseWeightDecay)
    print('LR:',learningRates[20])
    sgdconf = {learningRates = learningRates, momentum = momentum, weightDecays = weightDecays, learningRate = eta} 
    print('Start Iter: ', lastT)	
  
    numBatch = 20 --CHECK HERE!!!
    Channels = 3
    Length = 16
    Width = 112
    Height = 112
    local history1 = {}
    local history2 = {}
    logger = optim.Logger('Loss.log')
    logger:setNames{'Training Loss'}
    totalFile = io.open('log/total.log','w')
  
    --Allocating Cuda Memory
    input = torch.CudaTensor(numBatch, Channels, Length, Width, Height)
    target = torch.CudaTensor(numBatch) 
    logFileList = {}
  
    --Sample Data Line
    ln = 0 
    --Training Main Loop
  for t = lastT,opt.maxIter do    
    --Batches Loop
    for b = 1,numBatch do      
      --Going through samples
      ln = ln + 1
      if(ln > trainSamples) then        
        ln = 1
        --Learning Rate Decay Policy
        epoch = epoch + 1
        config.epoch = epoch
        eta = eta*math.pow(0.5,epoch/50)
        opt.eta = eta
        learningRates, weightDecays = model:getOptimConfig(eta, baseWeightDecay)
        sgdconf.learningRate = eta
        sgdconf.learningRates = learningRates
      end    
      --shuffled data
      lineNumber = idx[ln]         
      --Data File Path
      sampleLine = inputData[lineNumber][1]    
      --Data Label
      label = inputData[lineNumber][2]
      --Loading Data Using OpenCV
      cap  = cv.VideoCapture{inputData[lineNumber][1]}
      nF = cap:get{cv.CAP_PROP_FRAME_COUNT}     
      --Looping Frames
      for f=0,nF-1 do        
        cap:set{1,f}
        ret,frame = cap:read{}       
        frame = frame:permute(3,1,2)  --(Length, Width, Height)     
        --Looping Channels (For C3D Preparation)
        for c=1,Channels do          
          input[b][c][f+1] = (frame[c] - fmeans[c])--:div(255.0)          
        end --End of Channels Loop       
      end --End of Frames Loop
      cap:release()
      target[b] = label    
    end --End of Batches Loop
  
    --------------------------------------------------------------------
    -- define eval closure
    local feval = function(params)     
      model:zeroGradParameters()      
      local output = model:forward(input)
      _,pred_cat = torch.max(output:float(), 2)
      pred = pred_cat:float()
      label_gt = target:float()
      accuracy = pred:eq(label_gt):float():mean()
      j,i = torch.max(output[1],1)
      print('predicted:', i[1])
      print('ground-truth:', target[1])
      print('accuracy:', accuracy)
      local loss = criterion:forward(output,target)
      local  dloss_doutput = criterion:backward(output, target)
      model:backward(input, dloss_doutput)     
      return loss,grads
    end
    
    --Applying SGD
    _,fs = optim.sgd(feval, params, sgdconf)
    --Accumulating err over statIntervals
    err = err + fs[1]
    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t , opt.statInterval) == 0 then
      avgLoss = (err/opt.statInterval)
      print('==> iteration = ' .. t .. ', average loss = ' .. avgLoss .. ' lr '..eta )
      --file:write(err.."\n")
      logger:add{avgLoss}
      --logger:plot()
      table.insert(history1, {t, err})
      disp.plot(history1, {win=1, title='c3d_err', labels = {"iteration", "error"}})
      table.insert(history2, {t, accuracy})
      disp.plot(history2, {win=2, title='c3d_acc', labels = {"iteration", "accuracy"}})
      totalFile:write(t ..' '..avgLoss..''..label.."\n")
      err = 0     
    end
    
    model_name = '/home/gss/Code/Torch/C3DTorch/model/'
    if opt.save and math.fmod(t , opt.saveInterval) == 0 then
      model:clearState()
      model_f, err = io.open(model_name .. 'model_' .. t .. '.bin')
      if model_f == nil then 
        os.execute('cd.>model/model_' .. t .. '.bin')  end
      torch.save(model_name .. 'model_' .. t .. '.bin', model)
      config = {eta = eta, epsilon = epsilon, alpha = alpha, lastT = t ,iter = iter, epoch = epoch}
      config_f, err = io.open(model_name .. 'config_' .. t .. '.bin')
      if config_f == nil then 
        os.execute('cd.>model/config_' .. t .. '.bin')  end
      torch.save(model_name .. 'config_' .. t .. '.bin', config)
      
      --Saving Last Models for Power Failure Resume
      f, err = io.open(model_name .. 'model_Last.bin')
      if f == nil then 
        os.execute('cd.>model/model_Last.bin')   end
      torch.save(model_name .. 'model_Last.bin', model)
      
      f, err = io.open(model_name .. 'config_Last.bin')
      if f == nil then 
        os.execute('cd.>model/config_Last.bin')  end
      torch.save(model_name .. 'config_Last.bin', config)
      
      -- testing
      
      model = torch.load(model_name .. 'model_' .. t .. '.bin')--Loading Mean Data
      test_fmeans = torch.load('test_fmeans.dat')
      local test_input = torch.CudaTensor(1, 3, 16, 112, 112)   -- partial video description
      num = 0
      for i = 1, testSamples do
        print('Line: ', i, '/ ', testSamples)
        linenumber = test_idx[i]
        gt_label = test[linenumber][2]
        cap  = cv.VideoCapture{test[linenumber][1]}
        nF = cap:get{cv.CAP_PROP_FRAME_COUNT}  
        --Looping Frames
        for f=0,nF-1 do  
          cap:set{1,f}
          ret,frame = cap:read{}       
          frame = frame:permute(3,1,2)  --(Length, Width, Height)     
          --Looping Channels (For C3D Preparation)
          for c=1,3 do      
            test_input[1][c][f+1] = (frame[c] - test_fmeans[c])--:div(255.0)  #
          end --End of Channels Loop    
        end
        cap:release()
        predict = model:forward(test_input)
        _,pre_label = torch.max(predict,1)
        print('predict:', pre_label[1])
        print('groundtruth', gt_label)
        if (pre_label[1] == gt_label) then  num = num + 1 end
      end
      print('accuracy is :')
      print(num / testSamples)          
    end
  
  end -- End of Training Main Loop
    totalFile:close()
    print ('Training done')
    collectgarbage()    
  end -- End of Train Phase check
end
main()