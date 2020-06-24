using Flux: onecold

lsize = 0
nullt = (0,0,0)

function sample_diag_op()
    # returns 'i' for Hia, for i in 1,2,3
    op = onecold(rand(d, 1), labels)[1]
    return op
end

function DiagonalUpdate()
    # follow procedure in Notes.pdf

    # number of non-unity ops in sim cell
    n = sum(x -> x != 0, operator_list[:,1])

    # TODO: safely access 'n' outside of DiagonalUpdate function

    spin_prop = copy(spin_left)
    for m = 1:M
        # replace diag op with null and move on to next time slice
        if operator_list[m,1] in [1,2,3]
            rr = rand()
            # removal probability:
            # c1 defined in main
            P = min( (M-n+1)/β*c1, 1)
            if P > rr
                operator_list[m,1] = 0 
                # changing the location is not needed
                n = (n > 0) ? n-1 : 0
            end
            continue
 
        # replace null with diag op and move on to next time slice
        elseif operator_list[m,1] == 0 
            # decide whether to insert
            rr = rand()
            # insertion probability:
            # c1 defined in main
            P = min(β*c1/(M-n), 1)
            
            # insert? yes
            if P > rr
                # which one?
                operator_list[m,1] = sample_diag_op()
                
                # put it where?                
                # H_1a : diagonal site op
                if operator_list[m,1] == 1
                    n = n + 1 # increase expansion order
                    # now specify the site
                    site = rand(1:N)
                    operator_list[m,2] = site
                    operator_list[m,3] = site
                
                # H_2a : diagonal site op
                elseif operator_list[m,1] == 2                     
                    n = n + 1 # increase expansion order
                    num_up = sum(x -> x == 1, spin_prop)
                    num_dn = N - num_up 
                    insert_up = 3*Ω*num_up / (3*Ω*num_up + Ω*num_dn)

                    rr = rand()
                    if insert_up > rr
                        up_spins = findall(isodd, spin_prop) # spin=1 sites
                        site = up_spins[rand(1:num_up)] # choose spin=1 site
                        operator_list[m,2] = site
                        operator_list[m,3] = site

                    else
                        dn_spins = findall(iszero, spin_prop) # spin=0 sites
                        site = dn_spins[rand(1:num_dn)] # choose spin=0 site
                        operator_list[m,2] = site
                        operator_list[m,3] = site
                    end

                # H_3a : diagonal bond operator
                else
                    bond = rand(1:Nb) # choose a bond
                    spin1 = spin_prop[bond_spin[bond,1]]
                    spin2 = spin_prop[bond_spin[bond,2]]
 
                    if spin1 == spin2
                        n = n + 1 # increase expansion order
                        # site i = bond, site i+1 = bond+1
                        operator_list[m,2] = bond 
                        operator_list[m,3] = bond+1

                    else # reject the insertion
                        operator_list[m,1] = 0

                    end
                    # NOTE: Don't increase 'n' if spin1 != spin2, insertion fails! 
                end

            end
            continue

        # off-diagonal site operator flips spin
        elseif operator_list[i,1] == -1
            spin_prop[operator_list[i,2]] = xor(spin_prop[operator_list[i,2]],1)
       
        end
    end
end 

################################################################################

#function LinkedList() 
#    
#    global LinkList = zeros(Int,0)
#    global LegType = zeros(Int,0)
#    global Associates = []
#
#    First = zeros(Int,0)
#
#    for i = 1:N
#        push!(First,i)
#        push!(LinkList, -99)
#        push!(LegType, spin_left[i])
#        push!(Associates, nullt)
#    end
#
#    spin_prop = copy(spin_left)
#    
#    for i = 1:M
#        
#        if operator_list[i,1] == -1
#            site = operator_list[i,2]
#            push!(LinkList, First[site])
#            push!(LegType, spin_prop[site])
#            
