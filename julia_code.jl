function babyCT_B(dotx, x, p, t)

	# global xf, uf, Tf, T, A, B, M, R, Pt, per, amp, αa, αc, Quu, Qxu, Wcfinal, uDelay, x1Delay, x2Delay, x3Delay, x_save, t_save, uvec
	xf, T, Tf, A, B, M, R, Pt, percent, amplitude, αa, αc, u1Delay, u2Delay, u3Delay, x1Delay, x2Delay, x3Delay, x4Delay, x5Delay, x6Delay, pDelay, uvec = p#,i,maxIter,start = p # uu, vv useless here
	# println(t[1][1])
    n, m = 6,3
	Wc = x[(n+1) : Int(n+(n+m)*(n+m+1)/2)] # 4-13 # 
    Wa1 = x[52:57]#x[convert(Int64, (n+(n+m)*(n+m+1)/2+1)) : convert(Int64, (n+(n+m)*(n+m+1)/2+n))] # 14-16
	Wa2 = x[58:63]
	Wa3 = x[64:69]
	# P = x[end] # 17
	P = x[end] # 70

	# Update control
    ud = zeros(m,)
	ud[1] = Wa1'*(x[1:n] - xf)
	ud[2] = Wa2'*(x[1:n] - xf)
	ud[3] = Wa3'*(x[1:n] - xf)


	# Delays
    uddelay = zeros(m,)
	uddelay[1] = u1Delay(t - T)
	uddelay[2] = u2Delay(t - T)
	uddelay[3] = u3Delay(t - T)
    xdelay = zeros(n,)
	xdelay[1] = x1Delay(t - T)
    xdelay[2] = x2Delay(t - T)
    xdelay[3] = x3Delay(t - T)
	xdelay[4] = x4Delay(t - T)
	xdelay[5] = x5Delay(t - T)
	xdelay[6] = x6Delay(t - T)
	pdelay = pDelay(t - T)

	# Kronecker products
	U = vcat(x[1:n] - xf, ud[1], ud[2], ud[3]) # augmented state
    UkU = vcat(U[1]^2, U[1]*U[2], U[1]*U[3], U[1]*U[4], U[1]*U[5], U[1]*U[6], U[1]*ud[1], U[1]*ud[2], U[1]*ud[3],
               U[2]^2, U[2]*U[3], U[2]*U[4], U[2]*U[5], U[2]*U[6], U[2]*ud[1], U[2]*ud[2], U[2]*ud[3],
               U[3]^2, U[3]*U[4], U[3]*U[5], U[3]*U[6], U[3]*ud[1], U[3]*ud[2], U[3]*ud[3],
			   U[4]^2, U[4]*U[5], U[4]*U[6], U[4]*ud[1], U[4]*ud[2], U[4]*ud[3],
			   U[5]^2, U[5]*U[6], U[5]*ud[1], U[5]*ud[2], U[5]*ud[3],
			   U[6]^2, U[6]*ud[1], U[6]*ud[2], U[6]*ud[3],
               ud[1]^2, ud[1]*ud[2], ud[1]*ud[3],
			   ud[2]^2, ud[2]*ud[3],
			   ud[3]^2)
	UkUdelay = vcat(xdelay[1]^2, xdelay[1]*xdelay[2], xdelay[1]*xdelay[3], xdelay[1]*xdelay[4], xdelay[1]*xdelay[5], xdelay[1]*xdelay[6], xdelay[1]*uddelay[1], xdelay[1]*uddelay[2], xdelay[1]*uddelay[3],
               xdelay[2]^2, xdelay[2]*xdelay[3], xdelay[2]*xdelay[4], xdelay[2]*xdelay[5], xdelay[2]*xdelay[6], xdelay[2]*uddelay[1], xdelay[2]*uddelay[2], xdelay[2]*uddelay[3],
               xdelay[3]^2, xdelay[3]*xdelay[4], xdelay[3]*xdelay[5], xdelay[3]*xdelay[6], xdelay[3]*uddelay[1], xdelay[3]*uddelay[2], xdelay[3]*uddelay[3],
			   xdelay[4]^2, xdelay[4]*xdelay[5], xdelay[4]*xdelay[6], xdelay[4]*uddelay[1], xdelay[4]*uddelay[2], xdelay[4]*uddelay[3],
			   xdelay[5]^2, xdelay[5]*xdelay[6], xdelay[5]*uddelay[1], xdelay[5]*uddelay[2], xdelay[5]*uddelay[3],
			   xdelay[6]^2, xdelay[6]*uddelay[1], xdelay[6]*uddelay[2], xdelay[6]*uddelay[3],
               uddelay[1]^2, uddelay[1]*uddelay[2], uddelay[1]*uddelay[3],
			   uddelay[2]^2, uddelay[2]*uddelay[3],
			   uddelay[3]^2)		   
	# g(i,maxIter,4)
    Quu = [Wc[end-5] Wc[end-4] Wc[end-3]; Wc[end-4] Wc[end-2] Wc[end-1]; Wc[end-3] Wc[end-1] Wc[end]] # m x m
    Quu_inv = inv(Quu)
	Qux = [Wc[7] Wc[15] Wc[22] Wc[28] Wc[33] Wc[37]; Wc[8] Wc[16] Wc[23] Wc[29] Wc[34] Wc[38]; Wc[9] Wc[17] Wc[24] Wc[30] Wc[35] Wc[39]]
    #Qux = [Wc[5] Wc[10] Wc[14] Wc[17]; Wc[6] Wc[11] Wc[15] Wc[18]] # n x m
    Qxu = Qux' # m x n

	# Integral reinforcement dynamics
	dP = 0.5*(U[1:n]'*M*U[1:n] + ud'*R*ud)

	# approximation errors
	σ = UkU - UkUdelay
	σ_f	= UkU
    ec = P - pdelay + Wc'*σ
	ecfinal = 0.5*U[1:n]'*Pt*U[1:n] - Wc'*σ_f
    ea1 = Wa1'*U[1:n] + [Quu_inv[1,:]'*Qux[:,1]; Quu_inv[1,:]'*Qux[:,2]; Quu_inv[1,:]'*Qux[:,3]; Quu_inv[1,:]'*Qux[:,4]; Quu_inv[1,:]'*Qux[:,5]; Quu_inv[1,:]'*Qux[:,6]]'*U[1:n]
	ea2 = Wa2'*U[1:n] + [Quu_inv[2,:]'*Qux[:,1]; Quu_inv[2,:]'*Qux[:,2]; Quu_inv[2,:]'*Qux[:,3]; Quu_inv[2,:]'*Qux[:,4]; Quu_inv[2,:]'*Qux[:,5]; Quu_inv[2,:]'*Qux[:,6]]'*U[1:n]
	ea3 = Wa3'*U[1:n] + [Quu_inv[3,:]'*Qux[:,1]; Quu_inv[3,:]'*Qux[:,2]; Quu_inv[3,:]'*Qux[:,3]; Quu_inv[3,:]'*Qux[:,4]; Quu_inv[3,:]'*Qux[:,5]; Quu_inv[3,:]'*Qux[:,6]]'*U[1:n]


	# Critic dynamics
	dWc = -αc*((σ./(σ'*σ+1).^2)*ec + (σ_f./(σ_f'*σ_f+1).^2)*ecfinal)
	# g(i,maxIter,5)
	# Actor dynamics
    dWa1 = -αa*U[1:n]*ea1'
	dWa2 = -αa*U[1:n]*ea2'
	dWa3 = -αa*U[1:n]*ea3'

	# Persistence excitation
    unew = zeros(m,)
	# if t <= (p[10]/100)*p.Tf
	if t <= (percent/100)*Tf
	    # unew[1] = ud[1] + amplitude*exp(-.0005*t)*(sin(0.1*t)^2 + cos(0.7*t)^2) + sin(1.5*t)^2*cos(0.1*t) + sin(pi*t) + cos(0.1*t)
	    # unew[1] = ud[1] + p[11]*exp(-.0005*t)*(sin(0.1*t)^2+cos(0.7*t)^2)+sin(1.5*t)^2*cos(0.1*t)+sin(pi*t)+cos(0.1*t)
	    unew[1] = (ud[1]+amplitude*exp(-0.009*t)*2*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3))
	    unew[2] = (ud[2]+amplitude*exp(-0.009*t)*2*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3))
		unew[3] = (ud[3]+amplitude*exp(-0.009*t)*2*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3))

	else
	    unew .= ud
	end
	dx = A*U[1:n] + B*unew
	# g(i,maxIter,6)
	# p[end] = unew
	# p[end-1:end] .= unew
	uvec = [uvec; [unew]]
	# normESqvec = [normESqvec; 0]
	# trigCondvec = [trigCondvec; 0]
	# if (time() - start) > 2
	# 	# return dotx .= vcat(dx, dWc, dWa, dP)
	# 	return
	# end
	dotx .= vcat(dx, dWc, dWa1, dWa2, dWa3, dP)

end

function sim_TNNLS_B_CT_Local(x1, x2, S)#, goalVelocity) # x1 -> x2

	# System Dynamics for feedback
    n, m = 6, 3
	m_net = 10 # 10 - WAFR
	m_fuel = 30 # 30 - WAFR
	alpha = 0.05 # 0.05 - WAFR
	m_total = m_net + m_fuel # 40 - WAFR
	kx = 20 # 10 - CDC, 20 - WAFR
	ky = 20 # 10 - CDC, 20 - WAFR
	kz = 20 # 10 - CDC, 20 - WAFR
	cx = 45 # 10 - CDC, 45 - WAFR
	cy = 45 # 10 - CDC, 45 - WAFR
	cz = 45 # 10 - CDC, 45 - WAFR

    A = [0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; -kx/m_total 0 0 -cx/m_total 0 0; 0 -ky/m_total 0 0 -cy/m_total 0; 0 0 -kz/m_total 0 0 -cz/m_total] # n x n
    B = [0 0 0; 0 0 0; 0 0 0; 1/m_total 0 0; 0 1/m_total 0; 0 0 1/m_total] # n x m
    M = 10.0*Matrix(I, n, n) # n x n
    R = 2.0*Matrix(I, m, m) # m x m
	Pt = 0.5*Matrix(I, n, n)

	# check controllability & observability
    Co = ctrb(A, B)
    unco = size(A, 1) - rank(Co)
    # println("unco = $unco")
    Ob = obsv(sqrt(M), A)
    unob = size(sqrt(M), 1) - rank(Ob)
    # println("unob = $unob")

	if unco + unob > 0  error("system uncontrollable and/or unobservable")  end

    # ODE parametersYo
    Tf, T, N = 10, 0.05, 200 # finite horizon
    αc, αa = 90, 1.2
	amplitude, percent = 0.1, 50

	# start and goal
	xf = [x2; 0.0;0.0;0.0]
	x0 = [x1; S.lastVelocity]#0.0;0.0]

	# goal orientation
	theta = atan((x2-x1)[2]/(x2-x1)[1])

	# initialization
	#Wc0 = [10.0*ones(6); 5cos(theta+1); 5sin(theta+1); 5; 10.0*ones(5); 5cos(theta+1); 5sin(theta+1); 5; 10.0*ones(4); 5cos(theta+1); 5sin(theta+1); 5; 10.0*ones(3); 5cos(theta+1); 5sin(theta+1); 5; 10.0*ones(2); 5cos(theta+1); 5sin(theta+1); 5; 10.0; 5cos(theta+1); 5sin(theta+1); 5; 1; 0; 0; 1; 0; 1];
	#Wa10 = 5*cos(theta+1)*ones(n,) # n x m; 4 x 2
	#Wa20 = 5*sin(theta+1)*ones(n,)
	#Wa30 = 5*ones(n,)
	# initialization without angles
	Wc0 = [10.0*ones(6); 5; 5; 5; 10.0*ones(5); 5; 5; 5; 10.0*ones(4); 5; 5; 5; 10.0*ones(3); 5; 5; 5; 10.0*ones(2); 5; 5; 5; 10.0; 5; 5; 5; 1; 0; 0; 1; 0; 1];
	Wa10 = 5*ones(n,) # n x m; 4 x 2
	Wa20 = 5*ones(n,)
	Wa30 = 5*ones(n,)
    # Wc0 = [10.0*ones(Int((n+m)*(n+m+1)/2 - m*(m+1)/2), 1); 1; 0; 1][:] # (n+m)(n+m+1)/2 x 1; 10 x 1
	# Wa10 = 0.5*ones(n,) # n x m; 4 x 2
	# Wa20 = 0.5*ones(n,)

	u0 = [Wa10'*(x0 - xf); Wa20'*(x0 - xf); Wa30'*(x0 - xf)] # 5.0*ones(m,) # m x 1; 2 x 1

    t_save = [0,]
    u_save = [u0,] # u at every T
	x_save = [[x0; Wc0; Wa10; Wa20; Wa30; 0],]
	uvec = [u0,] # all u
    # println("u_save = $u_save")

    u1Delay = interp2PWC(getindex.(u_save, 1), -1, 1) # return an interpolation function
	u2Delay = interp2PWC(getindex.(u_save, 2), -1, 1)
	u3Delay = interp2PWC(getindex.(u_save, 3), -1, 1)
	x1Delay = interp2PWC(getindex.(x_save, 1), -1, 1)
    x2Delay = interp2PWC(getindex.(x_save, 2), -1, 1)
    x3Delay = interp2PWC(getindex.(x_save, 3), -1, 1)
	x4Delay = interp2PWC(getindex.(x_save, 4), -1, 1)
	x5Delay = interp2PWC(getindex.(x_save, 5), -1, 1)
	x6Delay = interp2PWC(getindex.(x_save, 6), -1, 1)
	pDelay = interp2PWC(getindex.(x_save, 70), -1, 1)

	# uvec1, uvec2 = 0, 0
	# trigCondvec = zeros(0)
	# normESqvec = zeros(0)
	xdist = norm(x0[1:3] - xf[1:3])
	error = 0.25
	# localKd = 0.0
	maxIter = 10000
	poseAndKd = Array{Tuple{Array{Float64,3},Float64}}(undef,0)

    # solve ODEs
	for i = 1:maxIter
		# global xf, uf, Tf, T, A, B, M, R, Pt, per, amp, αa, αc, Quu, Qxu, Wcfinal, uDelay, x1Delay, x2Delay, x3Delay, x_save, t_save, uvec
		t = ((i-1)*T, i*T)
		p = [xf, T, Tf, A, B, M, R, Pt, percent, amplitude, αa, αc, u1Delay, u2Delay, u3Delay, x1Delay, x2Delay, x3Delay, x4Delay, x5Delay, x6Delay, pDelay, uvec] #, normESqvec, trigCondvec]#, x_save, t_save, uvec1, uvec2]#, i,maxIter,start]
		# if i == NN println(1) end
        sol = solve(ODEProblem(babyCT_B, x_save[end], t, p), DP5())
		# if i == NN println(2) end
		t_save = [t_save; sol.t[2:end]] # vcat(t_save, sol.t) save time
		x_save = [x_save; sol.u[2:end]] # vcat(x_save, sol.u) save state
		u_save = [u_save; [uvec[end]]]
		# if i == NN println(3) end
		# println("sol.u[$i] = $(sol.u)")
		# println("sol.t[$i] = $(sol.t)")
		# uvec1 = p[end-1]
		# uvec2 = p[end]
		# println("uvec[$i] = $(p[end])")
		# if i == NN println(4) end
        u1Delay = interp2PWC(getindex.(u_save, 1), -1, i*T+.01) # interpolate control input
		u2Delay = interp2PWC(getindex.(u_save, 2), -1, i*T+.01)
		u3Delay = interp2PWC(getindex.(u_save, 3), -1, i*T+.01)
		x1Delay = interp2PWC(getindex.(x_save, 1), -1, i*T+.01)
		x2Delay = interp2PWC(getindex.(x_save, 2), -1, i*T+.01)
		x3Delay = interp2PWC(getindex.(x_save, 3), -1, i*T+.01)
		x4Delay = interp2PWC(getindex.(x_save, 4), -1, i*T+.01)
		x5Delay = interp2PWC(getindex.(x_save, 5), -1, i*T+.01)
		x6Delay = interp2PWC(getindex.(x_save, 6), -1, i*T+.01)
		pDelay = interp2PWC(getindex.(x_save, 70), -1, i*T+.01)
		# println(i)
		localKd = distPoint2Line(x_save[end][1:3], x1, x2)
		poseAndKd = [poseAndKd; ([x_save[end][1] x_save[end][2] x_save[end][3]], copy(localKd))]
		# if distPoint2Line(x_save[end][1:2], x1, x2) > kino_dist
		# 	kino_dist = distPoint2Line(x_save[end][1:2], x1, x2)
		# end

		# iter += 1

		# if euclidean(x_save[end][1:3], xf) < error*xdist break end
		if norm(x_save[end][1:3] - xf[1:3]) < error*xdist break end
	end
	S.lastVelocity = x_save[end][4:6]
	(poseAndKd, [0; 0; 0], [0; 0; 0])
	# (poseAndKd, normESqvec, trigCondvec)#,t_save)#eventTimes)
	# plot(getindex.(getindex.(poseAndKd, 1), 1), getindex.(getindex.(poseAndKd, 1), 2))
	# x_save_1 = getindex.(x_save, 1)
	# x_save_2 = getindex.(x_save, 2)
	# x_save_3 = getindex.(x_save, 3)
	# println("t_save = $t_save")
	# println("x1 = $(x_save_1[end])")
	# println("x2 = $(x_save_2[end])")
	# println("x3 = $(x_save_3[end])")
	# println("x1 = $(x_save[end][1])")
	# println("x2 = $(x_save[end][2])")
	# println("x3 = $(x_save[end][3])")
	# println("x4 = $(x_save[end][4])")
	# (path = [getindex.(x_save, 1) getindex.(x_save, 2)], kino_dist = kino_dist, lastVelocity = x_save[end][3:4])#, time = t_save)
	# [getindex.(x_save, 1) getindex.(x_save, 2) getindex.(x_save, 3) getindex.(x_save, 4)]
	# plot(getindex.(x_save, 1), getindex.(x_save, 2), width = 2, aspectratio = 1)
	# plot(t_save, getindex.(x_save, 1), width = 2)
	# plot!(t_save, getindex.(x_save, 2), width = 2)
	# plot!(t_save, getindex.(x_save, 3), width = 2)
	# plot!(t_save, getindex.(x_save, 4), width = 2)
	# elapsed = time() - start
end


# interp2PWC is a function that approximates the input vector data to a
# piecewise continuous function given the initial and final time
function interp2PWC(y, xi, xf)
    row = size(y, 1)
	if row == 1
	    xdata = range(-1.0, stop = xf, length = row + 1) # linspace in MATLAB; collect also works
		itp = interpolate([y[1];y[end]], BSpline(Cubic(Interpolations.Line(OnGrid()))))
	else
		xdata = range(xi, stop = xf, length = row)
		itp = interpolate(y, BSpline(Cubic(Interpolations.Line(OnGrid()))))
		# xdata = range(xi, stop = xf, length = 2)
		# itp = interpolate([y[1];y[end]], BSpline(Cubic(Line(OnGrid()))))
	end
	Interpolations.scale(itp, xdata)
end