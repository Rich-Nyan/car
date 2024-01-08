using DataFrames
using Plots

function plotter(str::AbstractString)
    T = 100
    iterations = [1]
    iter = length(iterations)
    players = 2

    reader = readlines(str)
    lines = length(reader)
    s = Vector{Float64}(undef,length(reader)*4)
    for i in 1:lines
        input = map(x -> parse(Float64, x), rsplit(reader[i],","))
        for j in 1:4
            s[4*(i-1)+j]=input[j]
        end
    end
    states = zeros(players,iter,T,4)
    for h in 1:players
        for i in 1:iter
            for j in 1:T
                for k in 1:4
                    states[h,i,j,k] = s[(i - 1) * (4 * T) + 4 * (j - 1) + k + (h - 1) * iter * T * 4]
                end
            end
        end
    end
    colors = palette(:default)[1:players]

    x_domain = extrema(states[:,:,:,1]) .+ (-0.5,0.5)
    y_domain = extrema(states[:,:,:,2]) .+ (-0.5,0.5)
    domain  = [minimum([x_domain[1],y_domain[1]]),maximum([x_domain[2],y_domain[2]])]

    gify = @animate for i=1:1:(iter*T)
        player = Int(ceil(i/T))
        timeframe = Int(i - (player - 1) * T)

        # Setup
        plot(
        linewidth = 4,
        label = false,
        xlabel = 'x',
        ylabel = 'y',
        title = "Self Driving Car",
        aspectratio = 1,
        ylimits = domain,
        xlimits = domain,
        legend = false
        )


        
        # Current Player

        # Trajectory
        for i in 1:players
            plot!(
                [states[i,player,k,1] for k in 1:1:timeframe],
                [states[i,player,k,2] for k in 1:1:timeframe],
                player,
                linewidth = 1,
                linecolor = colors[i],
                label = false
            )
            # Points
            scatter!(
                [states[i,player,timeframe,1]],
                [states[i,player,timeframe,2]],
                markersize = 5,
                color = colors[i]
            )
            annotate!(states[i,player,timeframe,1], states[i,player,timeframe,2], text(string(i), 8, :black))
        end


        
    end

    filename = split(str, "/")[end]
    filename = split(filename, ".")[1]
    gif(gify, "twoplayer_t1.gif", fps = 15)
end

plotter("2player_trajectory.txt")

