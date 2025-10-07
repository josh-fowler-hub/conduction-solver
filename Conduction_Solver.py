import numpy as np

"""
Simple 1D condudef implicit_heat_solver(
	T_inside,
	T_initial,
	length,
	thickness,
	total_time,
	T_inf,
	gasket_start,
	gasket_end,
	nx=1000,
	nt=10000,
	k=205,
	k_gasket=0.2,
	R_contact=5e-4,
	rho=2700,
	cp=900
): an aluminum wall.
Given inside temperature (T_in		for n in range(nt):
			b = T.copy()
			b[0] = T_inside
			# Convective BC: (	# Plot temperature profiles in Kelvin with different colors
	ax1.set_xlabel('Position (mm)')
	ax1.set_ylabel('Temperature (K)', color='black')
	ax1.plot(x_mm, T_K_max, color='red', linewidth=2, label='Max Temp (685 K)')
	ax1.plot(x_mm, T_K_min, color='blue', linewidth=2, label='Min Temp (410 K)')
	ax1.plot(x_mm, T_K_avg, color='green', linewidth=2, label='Avg Temp (599 K)')
	ax1.tick_params(axis='y', labelcolor='black')
	ax1.grid(True, alpha=0.3)

	# Add shaded region for gasket
	gasket_start_mm = gasket_start * 1000  # convert to mm
	gasket_end_mm = gasket_end * 1000      # convert to mm
	ax1.axvspan(gasket_start_mm, gasket_end_mm, color='lightgray', alpha=0.3, label='Gasket region')
	ax1.legend(loc='upper right')dx = h/k * (T_inf - T[-1])
			b[-1] = h*dx/k_array[-1] * T_inf, wall thickness, and heat flux,
calculates the outside wall temperature using Fourier's law.
"""

def conduction_solver(T_inside, thickness, heat_flux, k=205):
	"""
	Calculate the outside temperature of an aluminum wall.
	Args:
		T_inside (float): Inside wall temperature (C or K)
		thickness (float): Wall thickness (meters)
		heat_flux (float): Heat flux (W/m^2)
		k (float): Thermal conductivity of aluminum (W/m·K), default 205
	Returns:
		float: Outside wall temperature (same units as T_inside)
	"""
	T_outside = T_inside - (heat_flux * thickness) / k
	return T_outside

def implicit_heat_solver(
	T_inside,
	T_initial,
	length,
	thickness,
	total_time,
	T_inf,
	gasket_start,
	gasket_end,
	nx=1000,
	nt=10000,
	k=205,
	k_gasket=0.3,
	R_contact=5e-3,
	rho=2700,
	cp=900
):
	"""
	Implicit finite difference solver for 1D heat equation with variable thermal conductivity and contact resistance.
	Args:
		T_inside (float): Fixed inside wall temperature (boundary, K)
		T_initial (float): Initial temperature everywhere else (K)
		length (float): Wall length (meters)
		thickness (float): Wall thickness (meters)
		total_time (float): Total simulation time (seconds)
		T_inf (float): Freestream air temperature (K)
		gasket_start (float): Start position of gasket (meters)
		gasket_end (float): End position of gasket (meters)
		nx (int): Number of spatial grid points
		nt (int): Number of time steps
		k (float): Thermal conductivity of aluminum (W/m·K)
		k_gasket (float): Thermal conductivity of gasket (W/m·K)
		R_contact (float): Contact resistance at interfaces (m²·K/W)
		rho (float): Density (kg/m³)
		cp (float): Specific heat (J/kg·K)
	Returns:
		x (ndarray): Spatial grid
		T (ndarray): Temperature at final time
	"""
	# Create grid so gasket boundaries align with nodes
	# Calculate required nx to place gasket boundaries on nodes
	gasket_thickness = gasket_end - gasket_start
	
	# Use fine grid spacing for accuracy
	dx_fine = 0.0002  # 0.2 mm spacing
	
	# Calculate positions for key points
	n_start = int(gasket_start / dx_fine)
	n_end = int(gasket_end / dx_fine)
	n_total = int(thickness / dx_fine)
	
	# Adjust to ensure exact alignment
	gasket_start_adj = n_start * dx_fine
	gasket_end_adj = n_end * dx_fine
	thickness_adj = n_total * dx_fine
	
	nx = n_total + 1
	dx = dx_fine
	
	print(f"Adjusted gasket: {gasket_start_adj*1000:.3f} to {gasket_end_adj*1000:.3f} mm")
	print(f"Grid: nx={nx}, dx={dx*1000:.3f} mm")
	
	x = np.linspace(0, thickness_adj, nx)
	T = np.ones(nx) * T_initial
	T_old = T.copy()
	T[0] = T_inside  # Dirichlet BC at x=0

	# Define thermal conductivity array - gasket has different k
	k_array = np.ones(nx) * k  # aluminum by default
	gasket_mask = (x >= gasket_start_adj) & (x <= gasket_end_adj)
	k_array[gasket_mask] = k_gasket  # gasket material
	
	# Find interface nodes for contact resistance
	interface_start_idx = np.argmin(np.abs(x - gasket_start_adj))
	interface_end_idx = np.argmin(np.abs(x - gasket_end_adj))
	print(f"Gasket region: {np.sum(gasket_mask)} nodes from {gasket_start_adj*1000:.1f} to {gasket_end_adj*1000:.1f} mm")
	print(f"Interface nodes: {interface_start_idx} and {interface_end_idx}, R_contact = {R_contact} m²·K/W")

	# Build coefficient matrix A for implicit scheme with variable k and contact resistance
	alpha = k / (rho * cp)  # use aluminum properties for time step
	# Set r ~ 1 for accuracy
	r_target = 1.0
	dt = r_target * dx**2 / alpha
	nt = int(total_time / dt)
	r = alpha * dt / dx**2
	print(f"alpha = {alpha}, dx = {dx}, dt = {dt}, r = {r}, nt = {nt}")
	h = 25  # W/m^2·K (realistic for natural air convection)
	
	A = np.zeros((nx, nx))
	# Interior nodes with variable thermal conductivity and contact resistance
	for i in range(1, nx-1):
		# Check if this is an interface node with contact resistance
		if i == interface_start_idx or i == interface_end_idx:
			# Apply contact resistance at interface
			# Heat flux continuity: q = (T_left - T_right) / (R_contact + dx/(2*k_left) + dx/(2*k_right))
			if i == interface_start_idx:  # aluminum to gasket interface
				k_left = k
				k_right = k_gasket
			else:  # gasket to aluminum interface
				k_left = k_gasket
				k_right = k
			
			# Include contact resistance in thermal resistance
			R_total = R_contact + dx/(2*k_left) + dx/(2*k_right)
			conductance = 1.0 / R_total
			r_interface = conductance * dt / (rho * cp * dx)
			
			A[i, i-1] = -r_interface
			A[i, i] = 1 + 2*r_interface
			A[i, i+1] = -r_interface
		else:
			# Normal interior nodes
			# Use harmonic mean for thermal conductivity between nodes
			k_left = 2 * k_array[i-1] * k_array[i] / (k_array[i-1] + k_array[i])
			k_right = 2 * k_array[i] * k_array[i+1] / (k_array[i] + k_array[i+1])
			r_left = k_left * dt / (rho * cp * dx**2)
			r_right = k_right * dt / (rho * cp * dx**2)
			A[i, i-1] = -r_left
			A[i, i] = 1 + r_left + r_right
			A[i, i+1] = -r_right
	A[0, 0] = 1  # Dirichlet BC
	# Convective BC at x=L: use finite difference approximation
	# (T[-1] - T[-2])/dx = h/k * (T_inf - T[-1])
	A[-1, -2] = -1
	A[-1, -1] = 1 + h*dx/k_array[-1]  # use gasket k if at boundary
	print("Matrix A (last row):", A[-1])

	# Time stepping
	print("Initial temperature profile:", T)
	for n in range(nt):
		b = T.copy()
		b[0] = T_inside
		# Convective BC: (T[-1] - T[-2])/dx = h/k * (T_inf - T[-1])
		b[-1] = h*dx/k * T_inf
		try:
			T = np.linalg.solve(A, b)
		except np.linalg.LinAlgError as e:
			print("Matrix solve error:", e)
			return x, T
		if np.any(np.isnan(T)):
			print(f"NaN detected at time step {n}")
			break
		# Check for convergence (optional)
		if n % 100 == 0:
			print(f"Time step {n}, max temp: {T.max()}, min temp: {T.min()}")
			# Check for steady state (optional)
			if np.allclose(T, T_old, atol=1e-2):
				print("Steady state reached.")
				break
			T_old = T.copy()

	print("Final temperature profile:", T)
	return x, T

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	# Define parameters directly
	T_inside_max = 550.1      # Inside wall temperature (K)
	T_inside_min = 410.0
	T_inside_avg = 480.0
	T_initial = 300.0     # Initial temperature (K)
	thickness = 23.1648e-3    # Wall thickness (meters)
	total_time = 10000.0  # Total simulation time (seconds) - longer for steady state
	T_inf = 300.0         # Freestream air temperature (K)
	gasket_start = 5.8928e-3
	gasket_end = 17.272e-3
	x, Tmax = implicit_heat_solver(T_inside_max, T_initial, thickness, thickness, total_time, T_inf, gasket_start, gasket_end, k=205, k_gasket=0.2, R_contact=5e-4, rho=2700, cp=900)
	x, Tmin = implicit_heat_solver(T_inside_min, T_initial, thickness, thickness, total_time, T_inf, gasket_start, gasket_end, k=205, k_gasket=0.2, R_contact=5e-4, rho=2700, cp=900)
	x, Tavg = implicit_heat_solver(T_inside_avg, T_initial, thickness, thickness, total_time, T_inf, gasket_start, gasket_end, k=205, k_gasket=0.2, R_contact=5e-4, rho=2700, cp=900)

	# Convert units for plotting
	x_mm = x * 1000  # meters to mm
	x_in = x * 39.3701  # meters to inches
	T_K_max = Tmax  # assuming input is K
	T_F_max = Tmax * 9/5 - 459.67  # Kelvin to Fahrenheit
	T_K_max = Tmax  # assuming input is K
	T_F_max = Tmax * 9/5 - 459.67  # Kelvin to Fahrenheit
	T_K_min = Tmin  # assuming input is K
	T_F_min = Tmin * 9/5 - 459.67  # Kelvin to Fahrenheit
	T_K_avg = Tavg  # assuming input is K
	T_F_avg = Tavg * 9/5 - 459.67  # Kelvin to Fahrenheit

	fig, ax1 = plt.subplots(figsize=(10, 6))

	# Plot temperature profiles in Kelvin with different colors
	ax1.set_xlabel('Position (mm)')
	ax1.set_ylabel('Temperature (K)', color='black')
	ax1.plot(x_mm, T_K_max, color='red', linewidth=2, label=f'Max Temp ({T_inside_max} K)')
	ax1.plot(x_mm, T_K_min, color='blue', linewidth=2, label=f'Min Temp ({T_inside_min} K)')
	ax1.plot(x_mm, T_K_avg, color='green', linewidth=2, label=f'Avg Temp ({T_inside_avg} K)')
	ax1.tick_params(axis='y', labelcolor='black')
	ax1.grid(True, alpha=0.3)

	# Add shaded region for gasket instead of vertical lines
	gasket_start_mm = gasket_start * 1000  # convert to mm
	gasket_end_mm = gasket_end * 1000      # convert to mm
	ax1.axvspan(gasket_start_mm, gasket_end_mm, color='lightgray', alpha=0.3, label='Gasket region')
	ax1.legend(loc='upper right')

	# Top x-axis for inches
	ax_top = ax1.twiny()
	ax_top.set_xlim(ax1.get_xlim())
	ax_top.set_xticks(ax1.get_xticks())
	ax_top.set_xticklabels([f"{val/25.4:.2f}" for val in ax1.get_xticks()])
	ax_top.set_xlabel('Position (inches)')

	# Right y-axis for Fahrenheit
	ax2 = ax1.twinx()
	ax2.set_ylabel('Temperature (°F)', color='black')
	ax2.plot(x_mm, T_F_max, color='red', linestyle='dashed', alpha=0.7, linewidth=1)
	ax2.plot(x_mm, T_F_min, color='blue', linestyle='dashed', alpha=0.7, linewidth=1)
	ax2.plot(x_mm, T_F_avg, color='green', linestyle='dashed', alpha=0.7, linewidth=1)
	ax2.tick_params(axis='y', labelcolor='black')
	# ax2.grid(True)

	plt.title('Temperature Profile')
	plt.tight_layout()
	plt.savefig('conduction_profile.png')
# if __name__ == "__main__":
# 	# Example usage
# 	T_inside = float(input("Enter inside wall temperature (C or K): "))
# 	thickness = float(input("Enter wall thickness (meters): "))
# 	heat_flux = float(input("Enter heat flux (W/m^2): "))
# 	T_outside = conduction_solver(T_inside, thickness, heat_flux)
# 	print(f"Outside wall temperature: {T_outside:.2f}")
