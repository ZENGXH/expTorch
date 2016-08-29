local ucf101_helper = {}
-- read the train_list_split.txt and convert it into data_dict in lua format
-- @param video_list the txt file to be processed
-----------
-- classID_file = /data1/wangjiang/datasets/ucf101_list/ClassID_ucf101.txt
function ucf101_helper.ReadGTText2DataDict(
        classID_file, video_list, 
        classList, classes, classListVideo, classIndices,
        videoList, videoIndices, videoLabel, videoLength, videoPath)
    -- data_dict is wrap in 
    -- data_dict 
    --   'class_num': ...,
    --   'data': { 1: {'title': ,, , 'length': ,. ,'label':..},}
    
    assert(classID_file, 'classID_file not exist')
    assert(video_list, 'input_file not exist')
    local classID = io.open(classID_file, 'r')
    assert(classID ~= nil, classID_file..' broken')

    for class in classID:lines() do
        local index_class = #classes + 1
        print('push: ',index_class, class)
        classes[index_class] = class
        classIndices[class] = index_class
        classListVideo[index_class] = {}
    end
    classID:close()

    -- get video infomation
    print(string.format('open %s to read', video_list))
    local file = io.open(video_list, 'r')
    assert(file ~= nil, video_list..'not exist')
    local data = {}
    local data_dict = {}
    local length_list = {}
    local line_id = 1
    local runningIndex = 0
    for line in file:lines() do
        local index_class
        local index_video = line_id
        local title
        local label
        local length
        data[line_id] = {} -- create new vedio data
        -- parse the line
        local i = 1
        for w in string.gmatch(line, "%S+") do
            if i == 1 then
                title = w
            elseif i == 2 then
                length = tonumber(w)
            else 
                label = tonumber(w)
            end
            i = i + 1
        end
        index_class = label + 1
        data[line_id].label = w
        data[line_id].length = length
        data[line_id].title = w
        table.insert(videoList, title)
        videoIndices[title] = index_video
        videoLabel[title] = label
        table.insert(videoPath, title..'.bin')
        table.insert(videoLength, length)
        -- videoPath[title] = title..'bin'
        table.insert(classListVideo[index_class], title)
    end

    data_dict.video_num = #data
    data_dict.data = data
    print('[ReadGTText2DataDict] done, max_length ', torch.Tensor(videoLength):max(), 
                ' return data_dict in size, ', #videoLength)
    return data_dict
end

function ucf101_helper.MvFile(input_file, root, reduce_length, seg5_bin_path)
    local reduce_length = reduce_length or 0
    data_dict = ucf101_helper.ReadGTText2DataDict(input_file)
    video_num = data_dict['video_num']
    data = data_dict['data']
    local all_video_path = paths.concat(root, seg5_bin_path..'_video') 
    os.execute('mkdir '..all_video_path)
    for sample_id = 1, video_num do
        local title = data[sample_id]['title']
        -- print('make video path :', video_path)
        local video_path = paths.concat(root, seg5_bin_path, title)
        local length = data[sample_id]['length'] 
        local concat = {}
        concat.label = data[sample_id]['label']
        concat.index = sample_id
        concat.length = length
        concat.video = title
        concat.data = {}
        for frames_id = 0, length-1-reduce_length do
            local file_name = string.format('%s/%s/%s_%d_1views.bin', root, seg5_bin_path, title, frames_id) -- start from start_point: -1; filaname start from 0: -1 
            -- print(file_name)
            -- local file_name = string.format('%s/%s_%d_1views.bin', video_path, title, frames_id) 
            data_load = ucf101_helper.LoadBinFile(file_name)
            assert(data_load)
            table.insert(concat.data, data_load)
            -- local cmd = 'mv '..file_name..' .. '
            -- os.execute(cmd)
        end
        torch.save(paths.concat(all_video_path, title..'.bin'), concat)
        print('done ', sample_id )
    end
end

function ucf101_helper.LoadBinFile(filepath, datatype)
    local datatype = datatype or 'float'
    local file = torch.DiskFile(filepath, "r"):binary()
    local shape = {}
    for i = 1, 4 do
        local a = file:readInt(1)
        shape[i] = a[1]
    end
    local total = shape[1] * shape[2] * shape[3] * shape[4]
    
    local data
    if datatype == 'float' then
        data = file:readFloat(total)
    else 
        logger.fatal('not implement')
    end
    -- print('data', data)
    data = torch.FloatTensor(data)
    -- print('size of data: ', shape[1], shape[2], shape[3], shape[4])
    
    file:close()
    return data -- return 1024 feature if seg5. 101 feature if prob
end


print('load ucf101_helper done')
return ucf101_helper


