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
                # H0a : diagonal site op
                if operator_list[m,1] == 1
                    n = n + 1 # increase expansion order
                    # now specify the site
                    site = rand(1:N)
                    operator_list[m,2] = site
                    operator_list[m,3] = site
                
                # H1a : diagonal site op
                elseif operator_list[m,1] == 2  
                    println("Sampled H1a")

                    # check legs of H1a (i.e. spin up or down)
                    n = n + 1 # increase expansion order
                    num_up = sum(x -> x == 1, spin_prop)
                    num_dn = N - num_up 
                    insert_up = 3*Ω*num_up / (3*Ω*num_up + h*num_dn)

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

                # H2a : diagonal bond operator
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
        elseif operator_list[m,1] == -1
            spin_prop[operator_list[m,2]] = xor(spin_prop[operator_list[m,2]],1)
        end
    end
end 

################################################################################

function LinkedList() 
    
    len = 0 # Link list size

    for i in 1:M
        # site operator
        if operator_list[i,1] == -1 || operator_list[i,1] == 1 || operator_list[i,1] == 2
            len += 2
        elseif operator_list[i,1] == 3
            len += 4
        end
    end

    global LinkList = zeros(Int,len)
    global LegType = falses(len)
    global Associates = [nullt for _ in 1:len]
    global FlagForH1a = falses(len)

    #Need to 'mark' where H1a is in the sim. cell so that when clusters are
    #formed, we can determine if we pass through or not

    global First = zeros(Int, N)
    global Last = zeros(Int, N)
    idx = 0

    spin_prop = copy(spin_left)
    
    for i = 1:M
      
        # TODO: be aware that now clusters can 'overlap', though they shouldn't
 
        # H1a in Notes.pdf (i.e. diagonal site operator for long field)
        if operator_list[i,1] == 2 
            # TODO
            println("Somehow H1a appeared") 
            #lower / left leg
            site = operator_list[i,2]
            #@show First[site] 
            idx += 1
            LinkList[idx] = First[site]
            FlagForH1a[idx] = true
            LegType[idx] = spin_prop[site]
            current_link = idx

            if First[site] != 0
                LinkList[First[site]] = current_link
            else
                Last[site] = current_link
            end
            First[site] = current_link + 1

            # upper / right leg
            idx += 1
            LegType[idx] = spin_prop[site]
            FlagForH1a[idx] = true 
 
            
        # H0a in Notes.pdf (i.e. diag site operator for transverse field) 
        elseif operator_list[i,1] == 1 
           
            # lower / left leg
            site = operator_list[i,2]
            #@show First[site] 
            idx += 1
            LinkList[idx] = First[site]
            LegType[idx] = spin_prop[site]
            current_link = idx

            if First[site] != 0
                LinkList[First[site]] = current_link
            else
                Last[site] = current_link
            end
            First[site] = current_link + 1

            # upper / right leg
            idx += 1
            LegType[idx] = spin_prop[site]

 
        # H-1a in Notes.pdf (i.e. off-diag site operator)
        elseif operator_list[i,1] == -1

            # lower / left leg
            site = operator_list[i,2]
            #@show First[site]
            idx += 1
            LinkList[idx] = First[site]
            LegType[idx] = spin_prop[site]
            current_link = idx

            spin_prop[site] ⊻= 1 # spin flip

            if First[site] != 0
                LinkList[First[site]] = current_link
            else
                Last[site] = current_link
            end
            First[site] = current_link + 1

            # upper / right leg
            idx += 1
            LegType[idx] = spin_prop[site] 

        
        # H2a in Notes.pdf (i.e. diagonal bond operator)
        elseif operator_list[i,1] == 3

            site1, site2 = operator_list[i,2], operator_list[i,3]
            #@show First[site1]
            
            # lower left
            idx += 1
            LinkList[idx] = First[site1]
            LegType[idx] = spin_prop[site1]
            current_link = idx

            if First[site1] != 0
                LinkList[First[site1]] = current_link   
            else
                Last[site1] = current_link
            end

            First[site1] = current_link + 2
            vertex1 = current_link
            Associates[idx] = (vertex1 + 1, vertex1 + 2, vertex1 + 3)
           
            #@show First[site2]
 
            # lower right
            idx += 1
            LinkList[idx] = First[site2]
            LegType[idx] = spin_prop[site2]
            current_link = idx

            if First[site2] != 0
                LinkList[First[site2]] = current_link   
            else
                Last[site2] = current_link
            end
    
            First[site2] = current_link + 2
            Associates[idx] = (vertex1, vertex1 + 2, vertex1 + 3)

            # upper left
            idx += 1
            LegType[idx] = spin_prop[site1]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 3)
            
            # upper right
            idx += 1
            LegType[idx] = spin_prop[site2]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 2)
         
        end

    end

    # PBCs in imag. time
    for i in 1:N
        if First[i] != 0
            #@show i
            #@show First[i]
            #@show Last[i]
            LinkList[First[i]] = Last[i]
            LinkList[Last[i]] = First[i]
        end
    end

    # TODO:
    ## DEBUG
    #if spin_prop != spin_right
    #    println("Basis state propagation error in LINKED LIST")
    #end   

end

################################################################################

function ClusterUpdate()

    # TODO

    lsize = length(LinkList)

    # Given a simulation cell, let's say there are 3 total clusters to be made
    # labels clusters 1,2,3. 
    # in_cluster tells us which legs belong to which cluster.
    # e.g. in_cluster = [1,1,2,2,3,1,1,2,2]  

    global in_cluster=zeros(Int,lsize)

    cstack = zeros(Int,0)  #This is the stack of vertices in a cluster
    ccount = 0 #cluster number counter

    for i = 1:lsize

        #Add a new leg onto the cluster
        #only have to worry about matrix elements & operators that change
        #bond operators map to themselves and matrix elements don't change
        #site operators map to other site operators
        if (in_cluster[i] == 0 && Associates[i] === nullt)

            ccount += 1
            push!(cstack,i) 
            in_cluster[i] = ccount 

            # imagine approaching H1a FIRST. No matter what, the first leg will
            # be flipped (i.e. whether the cluster halts on H1a or passes
            # the entrance leg will still be flipped. So this code here does not
            # change 
            flip = rand(Bool) #flip a coin for the SW cluster flip
            if flip
                LegType[i] ⊻= 1 #spinflip
            end

            while !isempty(cstack)
                # leg = LinkList element.
                # e.g. cstack = [2], LinkList = [3,1,2,4]
                # leg = LinkList[pop!(cstack)] = 1
                # also note that after this, cstack = 0-element array
                tmp = pop!(cstack)
                flagH1a = FlagForH1a[tmp]
                leg = LinkList[tmp]
        
                #@show leg
           
     
                # this leg currently does NOT belong to a cluster
                if in_cluster[leg] == 0

                    # assign this leg to cluster 'ccount'
                    in_cluster[leg] = ccount; #add the new leg and flip it 
                    if flip 
                        LegType[leg] ⊻= 1 
                    end
                    
                    #@show flagH1a

                    if flagH1a
                        # decide whether or not to add the next vertex / leg to
                        # the same cluster
                        println("Somewhow H1a was flagged")
                        rr = rand()
                        P_pass = h / (Ω + h)

                        if P_pass > rr # don't halt. pass through
                            # assign this leg to cluster 'ccount'
                            next_leg = LinkList[tmp+1] 
                            #println("Legs ",leg," and ",next_leg," are in cluster ",ccount)
                            in_cluster[next_leg] = ccount; #add the new leg and flip it 
                            if flip 
                                LegType[next_leg] ⊻= 1 
                            end

                            #@show LegType[leg]
                            #@show LegType[next_leg]

                        end
                    end

                    #now check all associates and add to cluster
                    assoc = Associates[leg] #a 3-element array
                    if assoc != nullt

                        for a in assoc
                            push!(cstack, a)
                            in_cluster[a] = ccount
            
                            if flip
                                LegType[a] ⊻= 1
                            end
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
    spin_right = copy(spin_left)
    for i in 1:N
        if First[i] != 0 
            spin_left[i] = LegType[Last[i]]
            spin_right[i] = LegType[First[i]]
        else
            spin_left[i] = rand(0:1)
            spin_right[i] = spin_left[i]
        end
    end

    ocount = 1  # first leg
    for i = 1:M  
        #bond operator
        if operator_list[i,1] == 3
            ocount += 4
        #any of the site operators
        elseif operator_list[i,1] == 2 || operator_list[i,1] == 1 || operator_list[i,1] == -1
            if LegType[ocount] == LegType[ocount + 1] #diagonal
                # if it's a diagonal operator and H1a is there, don't need to 
                # change operator type
                if FlagForH1a[ocount]
                    println("Somehow H1a got flaaaaagggeedddd")
                end

                if !FlagForH1a[ocount]
                    operator_list[i,1] = 1
                    #println("DCHANGE ",i," ",ocount)
                end  
            else
                operator_list[i,1] = -1 #off-diagonal
                #println("OCHANGE ",i," ",ocount)
            end
            ocount += 2
        end
    end
end
