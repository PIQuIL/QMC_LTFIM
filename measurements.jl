function sample()
    
    sample_period = 151 # hard set for now
    spin_prop = copy(spin_left)
    samples = []    

    for i in 1:M
        if operator_list[i,1] == -1
            site = operator_list[i,2]
            spin_prop[site] ⊻= 1
        end

        if i%sample_period == 0        
            samples = isempty(samples) ? spin_prop : hcat(samples, spin_prop)
        end

    end

    return samples
end

function magnetization(samples)
    return sum(samples.*2 .-1) / (N * size(samples,2))
end

function energy()
    n = sum(x -> x != 0, operator_list[:,1])
    return -n/β
end

function Measure()
    #samples = sample()
    #return magnetization(samples)
    return energy()
end
