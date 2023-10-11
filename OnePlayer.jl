using Ipopt
using Optim
using LinearAlgebra
using ForwardDiff
using Plots
using JuMP

# Starts at time stamp 0

# x
# [1,5,9,...,4T-3]: x 
# [2,6,10,...,4T-2]: y 
# [3,7,11,...,4T-1]: theta 
# [4,8,12,...,4T]: velocity 

# u
# [1,3,5,...,2T-1]: theta
# [2,4,6,...,2T]: velocity

# Dynamics
# jacob : gets the derivative wrt x axis, y axis, angle, and velocity
# x : holds x,y,θ,v at that specific timestamp (state vector)
# u : theta and velocity

# Values
T = 100
dt = 0.1
initial_pose = [0,0,0,0]
final_pose = [4,4,pi/2,0]

z_guess = ones(10 * T + 4)

iterations = [1]
iter = length(iterations)

function dynamics!(jacob, x, u)
    jacob[1] = cos(x[3]) * x[4]
    jacob[2] = sin(x[3]) * x[4]
    jacob[3] = u[1]
    jacob[4] = u[2]
end

function objective_function(z)
    return z[4*T+1:6*T]'*z[4*T+1:6*T]
end
# Constraints h(z)
# h(z)
# T vectors of length 4 (x)
# T vectors of length 2 (u)
function dynamic_feasibility(z, model, T)
    x = z[1:4*T]
    u = z[4*T+1:end]

    h_z = zeros(QuadExpr, 4*T+4)
    # Nonlinearity auxillary variales
    cosine = @variable(model, [1:T])
    @NLconstraint(model, cosine[1] == cos(initial_pose[3]))
    @NLconstraint(model, [t = 2:T], cosine[t] == cos(x[4*t-5]))
    sine = @variable(model, [1:T])
    @NLconstraint(model, sine[1] == sin(initial_pose[3]))
    @NLconstraint(model, [t = 2:T], sine[t] == sin(x[4*t-5]))


    # Initial Pose 
    h_z[1] = (x[1] - (initial_pose[1] + dt * (cosine[1] * initial_pose[4])))
    h_z[2] = (x[2] - (initial_pose[2] + dt * (sine[1] * initial_pose[4])))
    h_z[3] = (x[3] - (initial_pose[3] + dt * u[1]))
    h_z[4] = (x[4] - (initial_pose[4] + dt * u[2]))

    # Intermediate Poses
    for i in 1:T-1
        h_z[4*i+1] = (x[4*i+1] - (x[4*i-3] + dt * (cosine[i + 1] * x[4*i])))
        h_z[4*i+2] = (x[4*i+2] - (x[4*i-2] + dt * (sine[i + 1] * x[4*i])))
        h_z[4*i+3] = (x[4*i+3] - (x[4*i-1] + dt * u[2*i+1]))
        h_z[4*i+4] = (x[4*i+4] - (x[4*i] + dt * u[2*i+2]))
    end

    # Final Pose
    h_z[4*T+1] = x[4*T-3] - final_pose[1]
    h_z[4*T+2] = x[4*T-2] - final_pose[2]
    h_z[4*T+3] = x[4*T-1] - final_pose[3]
    h_z[4*T+4] = x[4*T] - final_pose[4]

    return h_z
end

function lagrangian(z,λ, h_z)
    f_z = objective_function(z)
    return f_z + dot(λ, h_z)
end

function gradient_f(z)
    grad_f = zeros(QuadExpr, 10*T+4)
    for i in 4*T+1:6*T
        grad_f[i] == 2 * z[i]
    end
    return grad_f
end

function gradient_h(z,model)
    grad_h_z = zeros(QuadExpr,4*T+4,10*T+4)
    for i in 1:4*T+4
        for j in 1:10*T+4
            grad_h_z[i,j] = 0
        end
    end

    for i in 1:4*T
        grad_h_z[i,i] = 1
    end
    for i in 5:4*T
        grad_h_z[i,i-4] = -1
    end
    for i in 4T+1:4T+4
        grad_h_z[i,i-4] = 1
    end

    for i in 1:4*T
        if (i % 4 == 3)
            grad_h_z[i,4*T+2*div(i-3,4)+1] = dt
            grad_h_z[i+1,4*T+2*div(i-3,4)+2] = dt
        end
    end

    cosine = @variable(model, [1:T-1])
    @NLconstraint(model, [t = 1:T-1], cosine[t] == cos(z[4*t-1]))
    sine = @variable(model, [1:T-1])
    @NLconstraint(model, [t = 1:T-1], sine[t] == sin(z[4*t-1]))
    for i in 5:4*T
        if (i % 4 == 1)
            grad_h_z[i,i-2] = sine[div(i-1,4)] * z[i-1] * dt
            grad_h_z[i,i-1] = -cosine[div(i-1,4)] * dt
        end
        if (i % 4 == 2)
            grad_h_z[i,i-3] = -cosine[div(i-2,4)] * z[i-1] * dt
            grad_h_z[i,i-2] = -sine[div(i-2,4)] * dt
        end
    end

    return grad_h_z
end

#Optimize Function
function optimizer()
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5))

    @variable(model, z[i = 1:6 * T], start = z_guess[i])
    @variable(model, λ[i = 1:4*T+4], start = z_guess[6*T+i])

    h_z = dynamic_feasibility(z, model, T)

    # Objective
    @objective(model, Min, objective_function(z))

    # Lagrange gradient
    grad_f = gradient_f(z)
    grad_h = gradient_h(z, model)

    λ_quad = @NLexpression(model, sum(λ[i] * grad_h[j, i] for i in 1:4*T+4, j in 1:4*T+4))

    grad_f_quad = @NLexpression(model, sum(grad_f[i] for i in 4*T+1:6*T))
    
    @NLconstraint(model, norm(grad_f_quad + λ_quad) == 0)

    for i in 1:4*T+4
        @constraint(model,h_z[i] == 0)
    end
    states = zeros(iter, T, 4)
    for i in 1:iter 
        # Optimize the model
        optimize!(model)

        # Calculate h_z
        # z_guess = value.(z)
        for j in 1:T
            states[i, j, 1] = value(z[4*j-3])
            states[i, j, 2] = value(z[4*j-2])
            states[i, j, 3] = value(z[4*j-1])
            states[i, j, 4] = value(z[4*j])
        end
        
        println("Iteration $i:")
        println("Objective: ", objective_value(model))
        z_value = value.(z[6*T+1:end])
        println("Constraints: ", norm(z_value))
    end
    
    # Write the trajectory to a file
    open("trajectory.txt", "w") do io
        for i in 1:iter
            for j in 1:T
                println(io, states[i, j, 1], ",", states[i, j, 2], ",", states[i, j, 3], ",", states[i, j, 4])
            end
        end
    end
end

# Call the optimizer function
optimizer()
