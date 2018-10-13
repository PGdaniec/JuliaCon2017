# import mutable binary heap and fast random number generator
using DataStructures, RandomNumbers.Xorshifts,StatsBase, Random
import Base.getindex, Base.setindex!

# add getindex and setindex! for mutable binary heap
getindex(d::DataStructures.MutableBinaryHeap,i::Int64) = d.nodes[d.node_map[i]].value
setindex!(d::DataStructures.MutableBinaryHeap,v::Real,i::Int64) = update!(d,i,v)

# minimal code for spiking network simulation
function spikingnet(n,nspike,k,j0,ratewnt,tau,seedic,seedtopo)

    #neuron parameters (leaky integrate and fire neuron)
    iext     = tau*sqrt(k)*j0*ratewnt/1000 # iext given by balance equation
    w        = 1 /log(1. + 1/iext)
    c        = j0/sqrt(k)/(1. + iext) # phase velocity LIF
    phith    = 1.
    phishift = 0 .# threshold for LIF
    # init. random number generator (Xorshift)
    r = Xoroshiro128Star(seedic)
    #generate heap
    phi = mutable_binary_maxheap(rand(n))

    # initialize/preallocate spike raster and vector of receiving neurons
    spikeidx    = Int64[] #initialize time
    spiketimes  = Float64[] # spike raster
    postidx     = Array{Int64,1}(undef,k)

    # main loop
    @time for s = 1 : nspike
        # find next spiking neuron
        phimax, j = top_with_handle(phi)
        # calculate next spike time
        dphi = phith - phimax - phishift
        # global backshift instead of shifting all phases forward
        phishift += dphi
        # spiking neuron index is seed of rng to reduce memory
        Random.seed!(r,j+seedtopo)
        # sample receiving neuron index
        sample!(r,1:n-1,postidx)
        @inbounds for i = 1:k # avoid selfconnections
          postidx[i] >= j && ( postidx[i]+=1 )
        end
        # evaluate phase transition curve
        ptc!(phi,postidx,phishift,w,c)
        # reset spiking neuron
        phi[j]=-phishift
        # store spike raster
        push!(spiketimes,phishift) # store spiketimes
        push!(spikeidx,j) # store spiking neuron
    end
    #      rate                   , sidx    , stimes
    return nspike/phishift/n/tau*w, spikeidx, spiketimes*tau/w
end

#define phase transition curve
function ptc!(phi, postid, phishift, w, c)
    for i = postid
        phi[i] = - w*log(exp( - (phi[i] + phishift)/w) + c) - phishift
    end
end

n           = 10^5  #n: # of neurons
nspike      = 10^5
k           = 50    #k: synapses/neuron
j0          = 1     #j0: syn. strength
ratewnt     = 1.
tau         = .01   #tau: membr time const.
seedic      = 1
seedtopo    = 1

# compile code
spikingnet(100, 1, 10, j0, ratewnt, tau, seedic, seedtopo)

# gc()
rate,sidx,stimes = spikingnet(n,nspike,k,j0,ratewnt,tau,seedic,seedtopo)
@show rate
using PyPlot;
plot(stimes,sidx,",k",ms=0.1);
ylabel("Neuron Index",fontsize=20);xlabel("Time (s)",fontsize=20);
