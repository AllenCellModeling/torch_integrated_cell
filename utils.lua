
require 'torch'

utils = {}

function utils.ind2sub(matsize, ndx)
    
    matsize_tmp = torch.LongTensor(im_wl:size())
    for i = 1,im_wl:size():size() do
        matsize_tmp[i] = matsize[i]
    end
    
    matsize = matsize_tmp

    cp = torch.cumprod(matsize)

    sub = torch.zeros(matsize:size())
    
    for i = cp:size()[1], 2, -1 do
        vi = ((ndx-1) % cp[i-1]) + 1
        vj = (ndx - vi)/cp[i-1] + 1
        sub[i] = vj
        ndx = vi
    end
    sub[1] = ndx;
    
    return sub
end

function utils.alphanumsort(o)
--    Shamelessley lifted from 
--    http://notebook.kulchenko.com/algorithms/alphanumeric-natural-sorting-for-humans-in-lua
--     grj 5/16/16
  local function padnum(d) return ("%012d"):format(d) end
  table.sort(o, function(a,b)
    return tostring(a):gsub("%d+",padnum) < tostring(b):gsub("%d+",padnum) end)
  return o
end

function utils.meshgrid(x, y)
    local xx = torch.repeatTensor(x, y:size(1),1)
    local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
    return xx, yy
end

function utils.unique(input)

    local b = {}
    local range
    
    -- print(type(input))
    
    if type(input) == 'table' then
        range = #input 
    else
        range = input:numel()
    end
    
    local c = 0
    for i = 1, range do
        if b[input[i]] == nil then
            c = c+1
            b[input[i]] = c
        end
    end
    
    local u_vals = {}
    for i in pairs(b) do
        table.insert(u_vals,i)
    end
    
    local inds = torch.zeros(range)
    for i = 1, range do
        inds[i] = b[input[i]]
    end
    
    u_vals_tmp = {}
    
    for i = 1,#u_vals do
        ind = torch.nonzero(torch.eq(inds, torch.FloatTensor(inds:size()[1]):fill(i)))[1][1]
        u_vals_tmp[i] = input[ind]
        
    end
    u_vals = u_vals_tmp
        
    return u_vals, inds
end

function utils.table2float(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[orig_key] = utils.table2float(orig_value)
        end
        setmetatable(copy, utils.table2float(getmetatable(orig)))
    elseif orig_type == 'userdata' then -- number, string, boolean, etc
        copy = orig:float()
    end
    return copy
end

function utils.table2cuda(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[orig_key] = utils.table2cuda(orig_value)
        end
        setmetatable(copy, utils.table2cuda(getmetatable(orig)))
    elseif orig_type == 'userdata' then -- number, string, boolean, etc
        copy = orig:cuda()
    end
    return copy
end



function utils.split(inputstr, sep)
    -- liberated from http://stackoverflow.com/questions/1426954/split-string-in-lua
    if sep == nil then
        sep = "%s"
    end
    
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    
    return t
end

function utils.normalize(tensor)
    mu = torch.mean(tensor)
    std = torch.std(tensor)
    tensor_out = (torch.Tensor(tensor:size()):copy(tensor)-mu)/std
    
    return tensor_out
end

function utils.shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value:clone()
        end
    else -- number, string, boolean, etc
        copy = orig:clone()
    end
    return copy
end

function utils.deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[utils.deepcopy(orig_key)] = utils.deepcopy(orig_value)
        end
        setmetatable(copy, utils.deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig:clone()
    end
    return copy
end

