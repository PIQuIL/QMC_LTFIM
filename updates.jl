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

function LinkedList() 
    
    global LinkList = zeros(Int,0)
    global LegType = zeros(Int,0)
    global Associates = []

    First = zeros(Int,0)

    for i = 1:N
        push!(First,i)
        push!(LinkList, -99)
        push!(LegType, spin_left[i])
        push!(Associates, nullt)
    end

    spin_prop = copy(spin_left)
    
    for i = 1:M
      
        # TODO: be aware that now clusters can 'overlap', though they shouldn't
 
        # H1a in Notes.pdf
        if operator_list[i,1] == 2
            # TODO
             
 
        # H-1a in Notes.pdf
        elseif operator_list[i,1] == -1
            # TODO

            site = operator_list[i,2]                                            
            #lower or left leg                                                   
            push!(LinkList,First[site])                                          
            push!(LegType,spin_prop[site]) #the spin of the vertex leg           
            spin_prop[site] = xor(spin_prop[site],1) #spinflip                   
            current_link = size(LinkList)                                        
            LinkList[First[site]] = current_link[1] #completes backwards link    
            First[site] = current_link[1] + 1                                    
            push!(Associates,nullt)                                              
            #upper or right leg                                                  
            push!(LinkList,-99) #we don't yet know what this links to            
            push!(LegType,spin_prop[site]) #the spin of the vertex leg           
            push!(Associates,nullt)   

        # H0a in Notes.pdf
        elseif operator_list[i,1] == 1
            # TODO

            site = operator_list[i,2]                                                                                                     
            #lower or left leg                                                   
            push!(LinkList,First[site])                                            
            push!(LegType,spin_prop[site]) #the spin of the vertex leg             
            current_link = size(LinkList)                                          
            LinkList[First[site]] = current_link[1] #completes backwards link      
            First[site] = current_link[1] + 1                                        
            push!(Associates,nullt)                                                
            #upper or right leg                                                    
            push!(LinkList,-99) #we don't yet know what this links to            
            push!(LegType,spin_prop[site]) #the spin of the vertex leg             
            push!(Associates,nullt) 


        # H2a in Notes.pdf
        else 
            # TODO

            #lower left                                                              
            site1 = operator_list[i,1]                                             
            push!(LinkList,First[site1])                                             
            push!(LegType,spin_prop[site1]) #the spin of the vertex leg          
            current_link = size(LinkList)                                        
            LinkList[First[site1]] = current_link[1] #completes backwards link   
            First[site1] = current_link[1] + 2                                   
            vertex1 = current_link[1]                                            
            push!(Associates,(vertex1+1,vertex1+2,vertex1+3))                    
            #lower right                                                         
            site2 = operator_list[i,2]                                           
            push!(LinkList,First[site2])                                         
            push!(LegType,spin_prop[site2]) #the spin of the vertex leg          
            current_link = size(LinkList)                                        
            LinkList[First[site2]] = current_link[1] #completes backwards link   
            First[site2] = current_link[1] + 2                                   
            push!(Associates,(vertex1,vertex1+2,vertex1+3))                      
            #upper left                                                          
            push!(LinkList,-99) #we don't yet know what this links to            
            push!(LegType,spin_prop[site1]) #the spin of the vertex leg          
            push!(Associates,(vertex1,vertex1+1,vertex1+3))                      
            #upper right                                                         
            push!(LinkList,-99) #we don't yet know what this links to            
            push!(LegType,spin_prop[site2]) #the spin of the vertex leg          
            push!(Associates,(vertex1,vertex1+1,vertex1+2))   

        end

    end

    #The last N elements of the linked list are the final spin state            
    for i = 1:N                                                             
        push!(LinkList,First[i])                                                
        push!(LegType,spin_prop[i])                                             
        current_link = size(LinkList)                                           
        LinkList[First[i]] = current_link[1]                                    
        push!(Associates,nullt)                                                 
    end #i                                                                      
                                                                                
    global lsize = size(LinkList)                                               
    println("ABC", LinkList)                                                    
    #DEBUG                                                                      
    if spin_prop != spin_right                                                  
        println("Basis state propagation error: LINKED LIST")                   
    end   

end

################################################################################

function ClusterUpdate()

    # TODO

    #lsize is the size of the linked list
    in_cluster=zeros(Int,lsize[1])

    cstack = zeros(Int,0)  #This is the stack of vertices in a cluster

    ccount = 0 #cluster number counter
    for i = 1:lsize[1]

        #Add a new leg onto the cluster
        if (in_cluster[i] == 0 && Associates[i] == nullt)

            ccount+=1
            push!(cstack,i) 
            in_cluster[cstack[end]] = ccount  

            flip = rand(Bool) #flip a coin for the SW cluster flip
            if flip == true 
                LegType[cstack[end]] =  xor(LegType[cstack[end]],1) #spinflip
            end

            while isempty(cstack) == false

                leg = LinkList[cstack[end]]
                pop!(cstack)
                #println("leg ",leg," ",cstack)

                if in_cluster[leg] == 0

                    in_cluster[leg] = ccount; #add the new leg and flip it 
                    if flip == true 
                        #println("gonna flip ",leg)
                        LegType[leg] = xor(LegType[leg],1) 
                    end
                    #now check all associates and add to cluster
                    assoc = Associates[leg] #a 3-element array
                    if assoc != nullt
                        push!(cstack,assoc[1])
                        in_cluster[assoc[1]] = ccount
                        push!(cstack,assoc[2])
                        in_cluster[assoc[2]] = ccount
                        push!(cstack,assoc[3])
                        in_cluster[assoc[3]] = ccount
                        if flip == true 
                            LegType[assoc[1]] =  xor(LegType[assoc[1]],1) 
                            LegType[assoc[2]] =  xor(LegType[assoc[2]],1) 
                            LegType[assoc[3]] =  xor(LegType[assoc[3]],1) 
                        end
                    end #if
                end #if in_cluster == 0
            end #while
        end #if
    end #for i

    #DEBUG
    #for i = 1:lsize[1]
    #    println(i," ",LegType[i]," ",in_cluster[i])
    #end

    #map back basis states and operator list
    ocount = 0
    for i = 1:N
        spin_left[i] = LegType[i]  #left basis state
        ocount += 1
    end 

    ocount += 1  #next one is leg N + 1
    for i = 1:2*M  
        if operator_list[i,1] != -2 && operator_list[i,1] != -1
            ocount += 4
        else
            if LegType[ocount] == LegType[ocount + 1]  #diagonal
                operator_list[i,1] = -1
                #println("DCHANGE ",i," ",ocount)
            else
                operator_list[i,1] = -2 #off-diagonal
                #println("OCHANGE ",i," ",ocount)
            end
            ocount += 2
        end
    end

    for i = 1:N
        spin_right[i] = LegType[lsize[1] - N + i]  #left basis state
    end 

end
