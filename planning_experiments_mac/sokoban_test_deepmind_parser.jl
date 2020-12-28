using JLD
map = []
open("planning_experiments_mac/data_sokoban_test_deepmind.txt") do file
    linecount = 1
    for line in eachline(file)
        if linecount == 12
            linecount = 1
            println("Map done!")
            continue
            # break
        end

        if linecount > 1
            push!(map, line)
        end
        linecount += 1
    end
end

map_count = length(map) / 10
separate_maps = zeros(10,10,Int64.(map_count))

# floor = 1; wall = 2; agent = 3; goal = 4; box = 5
for i in 0:(map_count - 1)
    for j in (10*i + 1):(10*i + 10)
        line = map[Int64.(j)]
        line2num = zeros(1,length(line))
        for k in 1:length(line)
            if line[k] == '#'
                line2num[k] = 2
            elseif line[k] == ' '
                line2num[k] = 1
            elseif line[k] == '$'
                line2num[k] = 5
            elseif line[k] == '.'
                line2num[k] = 4
            elseif line[k] == '@'
                line2num[k] = 3
            end
        end

        separate_maps[Int64.(j - 10*i),:,Int64.(i+1)] = line2num
    end
end

data = separate_maps

save("planning_experiments_mac/data_sokoban_test_deepmind.jld","data", data)
