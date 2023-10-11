using DataFrames
using Plots

function plotter(str::AbstractString)
    T = 100
    iterations = [1]
    iter = length(iterations)

    reader = readlines(str)
    lines = length(reader)
    s = Vector{Float64}(undef,length(reader)*4)
    for i in 1:lines
        input = map(x -> parse(Float64, x), rsplit(reader[i],","))
        for j in 1:4
            s[4*(i-1)+j]=input[j]
        end
    end
    states = zeros(iter,T,4)
    for i in 1:iter
        for j in 1:T
            for k in 1:4
                states[i,j,k] = s[(i - 1) * (4 * T) + 4 * (j - 1) + k]
            end
        end
    end
    colors = palette(:default)[1:iter]

    x_domain = extrema(states[:,:,1]) .+ (-0.5,0.5)
    y_domain = extrema(states[:,:,2]) .+ (-0.5,0.5)
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

        # Previous Players
        if (player >= 2)
            for j in 1:player - 1
                plot!(
                [states[j,k,1] for k in 1:1:T],
                [states[j,k,2] for k in 1:1:T],
                j,
                linewidth = 1,
                linecolor = colors[j],
                label = false
                )
                scatter!(
                [states[j,T,1]],
                [states[j,T,2]],
                markersize = 5,
                color = colors[j],
                )
                annotate!(states[j,T,1], states[j,T,2], text(string(j), 8, :black))
            end
        end

        
        # Current Player

        # Trajectory
        plot!(
            [states[player,k,1] for k in 1:1:timeframe],
            [states[player,k,2] for k in 1:1:timeframe],
            player,
            linewidth = 1,
            linecolor = colors[player],
            label = false
        )


        # Points
        scatter!(
            [states[player,timeframe,1]],
            [states[player,timeframe,2]],
            markersize = 5,
            color = colors[player]
        )
        annotate!(states[player,timeframe,1], states[player,timeframe,2], text(string(iterations[player]), 8, :black))
    end

    filename = split(str, "/")[end]
    filename = split(filename, ".")[1]
    gif(gify, "t1.gif", fps = 15)
end

plotter("trajectory.txt")

