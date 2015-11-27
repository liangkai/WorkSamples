require('.')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing parameters')
cmd:text('Options')
cmd:option('-in_dim', 150, 'input dimension into lstm')
cmd:option('-mem_dim', 300,'memory dimension into lstm')
cmd:option('-num_classes', 4000, 'number of classes')
cmd:option('-batch_size', 33, 'batch_size')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- tests
include('tests/GradChecks.lua')
include('tests/TrainChecks.lua')
include('tests/CpuChecks.lua')
include('tests/GpuChecks.lua')

-- grad checks
grad_checker = imagelstm.GradChecks{}
grad_checker:check_lstm_captioner_hidden()
grad_checker:check_lstm_captioner()
grad_checker:check_add_module()
grad_checker:check_single_add_module()
grad_checker:check_concat_module()
grad_checker:check_concat_proj_module()

-- train checks
train_checker = imagelstm.TrainChecks{}
train_checker:check_train()

-- cpu checks
cpu_checker = imagelstm.CpuChecks{}
cpu_checker:check_cpu()

-- gpu checks
gpu_checker = imagelstm.GpuChecks{in_dim = params.in_dim, 
									mem_dim = params.mem_dim, 
									num_classes = params.num_classes}
gpu_checker:check_gpu()

