#!/usr/local/bin/python3
from __future__ import division
import argparse
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator, CubicHermiteSpline

start = time.time()

plt.rcParams['axes.formatter.useoffset'] = False # Remove axis offsetting in matplotlib - Makes plots look nicer
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

parser=argparse.ArgumentParser()
parser.add_argument('mode',type=str,help="Mode of operation. Specify 'plot_F', 'plot_G', 'plot_G_normalise', 'plot_pressures', or 'plot_pressures_normalise'.")
parser.add_argument('-ylim',nargs='+',type=float,help="Manually supply the lower and upper y-axis limits in units of eV.")
parser.add_argument('-xlim',nargs='+',type=float,help="Manually supply the lower and upper x-axis limits in the relevant pressure units (i.e. GPa or TPa).")
parser.add_argument('-figsize',nargs=2,type=float,default=[6,4.5],help="Matplotlib figure size.")
parser.add_argument('-give_errors',action='store_true',help="If this flag is supplied, the shaded error widths will be removed from all curves.")
parser.add_argument('-guess_params',nargs='+',type=float,help="Initial guess for EoS parameters E_O, V_0, B_0 and B_0_prime (eV, Ang^3, GPa, dimensionless respectively) as space-separated list.")
parser.add_argument('-normalising_structure',default=None,type=str,help="For the 'plot_G_normalise' or 'plot_pressures_normalise' modes , the data will be normalised to this structure.")
parser.add_argument('-eos',default=None,type=str,help="Analytical equation of state to be fitted to. 'Vinet','BM' or 'PT'.")
parser.add_argument('-interpolator',default='CubicSpline',help="Type of scipy interpolator. Choose from 'CubicSpline', 'Akima1DInterpolator', 'PchipInterpolator' or 'CubicHermiteSpline' ")
parser.add_argument('-extrapolation_factor',default=0.05,type=float,help="""Factor by which to extrapolate the volume in the F vs V curve. Setting this
to zero means no extrapolation at all. The volumes will become: (min_volume)*(1-extrapolation_factor) and (max_volume)*(1+extrapolation_factor).""")
parser.add_argument('-functional',help='Optionally pass the name of the functional used for the calculations - This will be used as a label on the plots.')
parser.add_argument('-give_minima',action='store_true',help="""If this flag is supplied when using the 'plot_G_normalise' mode, will print out
the smallest values of ΔG for each curve (relative to the normalising structure).""")
parser.add_argument('-give_maxima',action='store_true',help="""If this flag is supplied when using the 'plot_G_normalise' mode, will print out
the largest values of ΔG for each curve (relative to the normalising structure).""")
parser.add_argument('-use_derived_volume',action='store_true',help="""If this flag is supplied, plotted volumes will be derived from the
(numerical) gradient of G. Default behaviour is to fit the DFT volumes & pressures to a cubic spline.""")
parser.add_argument('-tpa',action='store_true',help="If this flag is supplied, pressure units on plots will be TPa.")
parser.add_argument('-filter_outliers',action='store_true',help="If this flag is supplied, raw datapoints outside the normalised range will be removed.")
parser.add_argument('-legend_loc',type=int,help="Matplotlib legend location (integer, 0-10).")
parser.add_argument('-legend_ncols',type=int,help="Matplotlib number of columns in legend.")
parser.add_argument('-shade_stability',action='store_true',help="""If this flag is supplied, the lowest enthalpy structures will be shaded in their stability region,
	provided that these have been specified in the .pressure_vary files via use of the '#shadelow=' and '#shadehigh=' tags.""")
parser.add_argument('-label_structures',action='store_true',help="If this flag is supplied, the lowest enthalpy structures will be labelled with their structure name.")
parser.add_argument('-num_fine_points',type=int,default=10**6,help="""Number of points in the fine (interpolated) grid. Use 10^5 or 10^6 for publication quality.
Can converge residual fitting errors wrt to num_fine_points (stops changing after like 10^6 or so).""")
parser.add_argument('-no_enthalpy_scatter',action='store_true',help="If this flag is supplied, DFT enthalpies will not be plotted as scatter points.")
args=parser.parse_args()

mode=args.mode
ylim=args.ylim
xlim=args.xlim
figsize=args.figsize
give_errors=args.give_errors
guess_params=args.guess_params
normalising_structure=args.normalising_structure
eos=args.eos
interpolator=args.interpolator
extrapolation_factor=args.extrapolation_factor
functional=args.functional
give_minima=args.give_minima
give_maxima=args.give_maxima
use_derived_volume=args.use_derived_volume
tpa=args.tpa
filter_outliers=args.filter_outliers
legend_loc=args.legend_loc
legend_ncols=args.legend_ncols
label_structures=args.label_structures
shade_stability=args.shade_stability
num_fine_points=args.num_fine_points
no_enthalpy_scatter=args.no_enthalpy_scatter

####################################################################################################
# Globally useful stuff, conversion factors, etc

cwd = os.getcwd()
file_list=os.listdir("%s" % (cwd))

charge_e = 1.602176634e-19 # electronic charge e
volume_conversion_factor = (charge_e * 10**21) # with V = dG/dp in units of eV / GPa, multiply by this to convert the volume to Ang^3
if tpa: volume_conversion_factor = volume_conversion_factor / 1000 # Ditto for TPa

# If no legend location or ncols was specified, use these as defaults
if not legend_loc:
	legend_loc = 1
if not legend_ncols:
	legend_ncols = 3

guess_params_supplied=False
if guess_params is not None:
	guess_params_supplied=True
	guess_params[2] = guess_params[2] / 160.21766208 # User supplies Bulk Modulus in GPa, so convert this to eV / A^3

####################################################################################################
# Parse and set some universal stuff

# Set pressure units
pressure_units = 'GPa'
if tpa:
	pressure_units = 'TPa'

# Check whether user has requested a spline fit, or an equation of state fit
use_spline = True
if eos:
	use_spline = False

# Parse scipy interpolator type (hereafter, 'interpolator' will be a callable)
if interpolator == "CubicSpline":
	interpolator = CubicSpline
elif interpolator == "Akima1DInterpolator":
	interpolator = Akima1DInterpolator
elif interpolator == "PchipInterpolator":
	interpolator = PchipInterpolator
elif interpolator == "CubicHermiteSpline":
	interpolator = CubicHermiteSpline
else:
	print("ERROR: Interpolator type not recognised.")
	print("Should be one of: 'CubicSpline', 'Akima1DInterpolator', 'PchipInterpolator' or 'CubicHermiteSpline' ")
	exit()

####################################################################################################
# Equations of State functions & derivatives (Vinet, Birch-Murnaghan (BM) and Poirier-Tarantola (PT)).
# Have tested all the analytical derivatives as correct using finite differences.

def f_eos(V,E_0,V_0,B_0,B_0_prime):
	V_0=abs(V_0)
	if eos == 'Vinet':
		E = E_0 + (2*B_0*V_0)/(B_0_prime-1)**2 * ( 2 - (5+3*abs(V/V_0)**(1/3)*(B_0_prime-1)-3*B_0_prime)*np.exp(-(3/2)*(B_0_prime-1)*(abs(V/V_0)**(1/3)-1)) )
	elif eos == 'BM':
		E = E_0 + 9*V_0*B_0/16 * ( (abs((V_0/V))**(2/3)-1)**3*B_0_prime + (abs((V_0/V))**(2/3)-1)**2 * (6-4*abs((V_0/V))**(2/3))  )
	elif eos == 'PT':
		E = E_0 + (B_0*V_0)/2 * np.log(V_0/V)**2 * ( 1 + (B_0_prime-2)/3 * np.log(V_0/V) )
	elif eos == 'Tait':
		E = E_0 + (B_0*V_0)/(B_0_prime+1) * ( (V/V_0) - 1 + 1/(B_0_prime+1) * ( np.exp( (B_0_prime+1)*(1-V/V_0) ) -1 ) )
	return E

def df_dV_eos(V,E_0,V_0,B_0,B_0_prime):
	V_0=abs(V_0)
	if eos == 'Vinet':
		df_dV = (2*B_0*V_0)/(B_0_prime-1)**2 * -1/(V**(2/3)*V_0**(1/3))*(B_0_prime-1) * ( 1 - 0.5*( 5 + 3*(V/V_0)**(1/3)*(B_0_prime-1) - 3*B_0_prime ) ) * np.exp(-3/2*(B_0_prime-1)*((V/V_0)**(1/3)-1))
	elif eos == 'BM':
		df_dV = 9*V_0*B_0/16 * (V_0)**(2/3)/V**(5/3) * ( -2*( (V_0/V)**(2/3) -1 )**2 * B_0_prime - (4/3)*( (V_0/V)**(2/3) -1 )*( 6-4*(V_0/V)**(2/3) ) + (8/3)*( (V_0/V)**(2/3) -1 )**2  )
	elif eos == 'PT':
		df_dV = -(B_0*V_0)/(2 * V) * np.log(V_0/V) * ( 2 + (B_0_prime-2) * np.log(V_0/V) )
	elif eos == 'Tait':
		df_dV = B_0/(B_0_prime+1) * ( 1-np.exp( (B_0_prime+1)*(1-V/V_0) ) )
	return df_dV

def d2f_dV2_eos(V,E_0,V_0,B_0,B_0_prime):
	V_0=abs(V_0)
	if eos == 'Vinet':
		x=(V/V_0)**(1/3)
		d2f_dV2 = (B_0/V_0) * (1/x)**(4) * np.exp( (3/2)*(B_0_prime-1)*(1-x) ) * ( 1 + 2*(1-x)/x + (3/2)*(B_0_prime-1)*(1-x) )
	elif eos == 'BM': # THIS IS WRONG, fix at some point
		d2f_dV2 = 9*V_0*B_0/16 * ( ((10/9)*((V_0)**(2/3)/(V)**(8/3))-1)**3*B_0_prime + ((10/9)*((V_0)**(2/3)/(V)**(8/3))-1)**2 * (6-(40/9)*((V_0)**(2/3)/(V)**(8/3)))  )
	elif eos == 'PT':
		d2f_dV2 = (B_0*V_0)/(2 * V**2) * ( np.log(V_0/V) * ( 2 + 2*(B_0_prime-2) + (B_0_prime-2)*np.log(V_0/V) ) + 2 )
	elif eos == 'Tait':
		d2f_dV2 = (B_0/V_0**2) * np.exp( (B_0_prime+1)*(1-V/V_0) )
	return d2f_dV2

def pV_eos(V,V_0,B_0,B_0_prime):
	V_0=abs(V_0)
	if eos == 'Vinet':
		x=(V/V_0)**(1/3)
		p=3*B_0 * ((1-x)/x**2) * np.exp(3/2 * (B_0_prime-1)*(1-x))
	return p

####################################################################################################
# Define a 'trim_data' function that trims supplied data between two pressures.
# This is called if the user specifies either '#trimlow=X' and/or '#trimhigh=Y' in the data file.

def trim_data(data_grid,pressure_grid,p_min,p_max):
	if p_min is None:
		p_min = -1E20
	if p_max is None:
		p_max = 1E20
	zipped_data = zip(data_grid,pressure_grid)
	zipped_data = np.array([ x for x in zipped_data if ( (x[1]>p_min) and (x[1]<p_max) ) ])
	data_grid = zipped_data[:,0]
	pressure_grid = zipped_data[:,1]
	return(data_grid,pressure_grid)

################################################################################################################
# Initialise subplots, define a marker tray, define a function to parse structure names.

fig, ax = plt.subplots(figsize=figsize)

marker_tray = ['o','s','v','^','D','p','P','*','<','>','h','8']

def parse_structure_name(structure_name):
	if 'sl' in structure_name:
		structure_name = structure_name.replace('sl','/')
	if 'un' in structure_name:
		structure_name = structure_name.replace('un','_')
	if '-' in structure_name:
		to_be_barred=structure_name.split('-')[-1][0] # this character will have a bar over it
		structure_name = structure_name.replace('-'+to_be_barred,'\\overline{%s}'%(to_be_barred)) # 'overline' is overbar for LaTex
	return structure_name

################################################################################################################
# User printouts, errors etc.

if len(file_list) == 0:
	print('No <name>_pressure_vary.results files found. Aborting.')
	exit()

if  ( ( (mode == 'plot_G_normalise') or (mode == 'plot_pressures_normalise') ) and (normalising_structure+'_pressure_vary.results' in file_list) ): # If the normalising structure file is there, bring it to the front of the list. Energies will be normalised to this structure.
	file_list.insert( 0, file_list.pop(file_list.index(normalising_structure+'_pressure_vary.results')) )
elif mode == 'plot_G_normalise':
	print('Specified normalising structure results file (%s_pressure_vary.results) not found in current directory. Aborting.' % (normalising_structure))
	exit()

################################################################################################################

count=0 # Structure index counter

if normalising_structure:
	normalising_structure = parse_structure_name(normalising_structure)

for filename in file_list:
	if filename.endswith("pressure_vary.results"):
		file=open(cwd+'/'+filename,'r')
		lines=file.readlines()
		file.close()
		# Get the structure name by splitting the filename on '_' (structure name is always first element)
		structure_name=filename.split("_")[0]
		# Dealing with underscores and slashes in spacegroup names...
		structure_name = parse_structure_name(structure_name)
		print(structure_name)
		extrapolation_factor_lower=extrapolation_factor # Will remain as this unless overridden with a user-specified '#extrapolation_factor=X' in the file.
		extrapolation_factor_higher=extrapolation_factor # Ditto.
		# The following list will hold the results as we parse them from the file
		results=[]
		# If any are supplied in the results file, the following list will be filled with tags (lines starting with a '#').
		# Tags indicate special things, i.e. imaginary modes beyond a certain pressure or the type of functional used. See later in code.
		tags=[]
		for line in lines:
			if '#' not in line:
				line.replace(' ','') # Remove all whitespace
				linesplit = line.split(",") # Data is comma-delimited; split each line on a comma
				results.append(linesplit)
			else:
				tag = line.replace(' ','').lower() # Remove all whitespace and cast to lowercase
				tags.append(tag) # If there's a '#' in the line, this is some sort of tag and will be parsed later.
				if '#extrapolation_factor_lower' in tag: # If present, process this particular tag right now
					extrapolation_factor_lower=float(tag.split("=")[-1])
				if '#extrapolation_factor_higher' in tag: # If present, process this particular tag right now
					extrapolation_factor_higher=float(tag.split("=")[-1])
		results=sorted(results,key=lambda x: float(x[0]))
		results_array=np.array(results) # Cast to array
		 # Cast all data to float & sort by pressure
		DFT_pressures=[ float(x) for x  in results_array[:,0] ]
		volume_values=[ float(x) for x  in results_array[:,1] ]
		helmholtz_energy=[ round(float(x),7) for x  in results_array[:,2] ]
		try:
			enthalpies=[ round(float(x),7) for x  in results_array[:,3] ]
		except:
			enthalpies = None
		vol_max, vol_min = volume_values[0], volume_values[-1]
		# If volume is negative, user has probably got energies and volumes wrong way round in results file...
		if any(vol < 0 for vol in volume_values):
			print('Error. Detected a negative volume - Have you got energies and volumes wrong way round in results file?')
			print('  -> Check ordering of columns in %s_pressure_vary.results file, it should be: <pressure>,<volume>,<energy> or <pressure>,<volume>,<energy>,<DFT enthalpy>' % (structure_name))
			exit()
		################################################################################################################
		# Get colour from cycle
		num = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])
		colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][count%num]
		################################################################################################################
		# Fitting...
		volume_values.sort()
		helmholtz_energy.reverse()
		volume_values_fine = np.linspace(vol_min*(1-extrapolation_factor_lower),vol_max*(1+extrapolation_factor_higher),num_fine_points)

		# If user requested a spline fit
		if use_spline:
			F_CS = interpolator(volume_values,helmholtz_energy,extrapolate=True)
			helmholtz_values_fitted = F_CS(volume_values_fine)
			if use_derived_volume: # Then the enthalpy / Gibbs free energy is to be calculated via a derivative
				fine_pressures = F_CS.derivative()
				fine_pressures = -fine_pressures(volume_values_fine)
				gibbs_free_energy = helmholtz_values_fitted + fine_pressures * volume_values_fine
				pressure_values_gpa_fine = [ (x * 160.21766208) for x in fine_pressures ] # Convert to GPa
			else: # Then we'll fit the actual DFT pressures & enthalpies to a (different) spline
				if (enthalpies is None):
					print("ERROR: Cannot fit enthalpies because none were found in %s_pressure_vary.results file" % (structure_name))
					print("To derive the enthalpies via the derivative of the energy, supply the 'use_derived_volume' flag")
					exit()
				H_CS = interpolator(DFT_pressures,enthalpies,extrapolate=True) # Cubic spline for H
				pressure_values_gpa_fine = np.linspace(DFT_pressures[0]*(1-extrapolation_factor_lower),DFT_pressures[-1]*(1+extrapolation_factor_higher),num_fine_points)
				gibbs_free_energy = H_CS(pressure_values_gpa_fine)

		# Otherwise, user requested an eos fit
		else:
			guess_params=[ helmholtz_energy[0],int(volume_values[0]),0.3,4 ] # Use first F & V values as initial guesses for E_0 and V_0, and guess B_0=0.3, B_0_prime=4
			params, covariances = curve_fit(f_eos,volume_values,helmholtz_energy,guess_params,maxfev=10**8)
			E_0_error=np.sqrt(covariances[0][0])
			V_0_error=np.sqrt(covariances[1][1])
			B_0_error=np.sqrt(covariances[2][2])
			B_0_prime_error=np.sqrt(covariances[3][3])
			E_0_fitted=float(params[0])
			V_0_fitted = float(params[1])
			B_0_fitted = float(params[2])
			B_0_prime_fitted = float(params[3])
			helmholtz_values_fitted = [ f_eos(x,E_0_fitted,V_0_fitted,B_0_fitted,B_0_prime_fitted) for x in volume_values_fine ]
			################################################################################################################
			# Calculate pressure and Gibbs free energy
			pressure_values_fine = [ - df_dV_eos(x,E_0_fitted,V_0_fitted,B_0_fitted,B_0_prime_fitted) for x in volume_values_fine ]
			gibbs_free_energy = np.array([ (helmholtz_values_fitted[j] + pressure_values_fine[j]*volume_values_fine[j]) for j in range(len(pressure_values_fine)) ])
			pressure_values_gpa_fine = [ (x * 160.21766208) for x in pressure_values_fine ] # convert to GPa

		################################################################################################################
		# Convert to terapascal, if requested
		if tpa:
			pressure_values_gpa_fine = np.array(pressure_values_gpa_fine) / 1000 # Convert to TPa
			DFT_pressures = np.array(DFT_pressures) / 1000 # Convert to TPa

		################################################################################################################
		# Calculate errors, if requested

		# Error in F
		if give_errors:
			residual_datapoints = [ np.argmin ( np.abs(np.array(volume_values_fine) - volume_values[j] ) ) for j in range (len(volume_values)) ] # Finds the datapoints in volume_values_fine that are closest to the actual volume datapoints (they should be very close)
			if use_spline:
				residuals = [ F_CS(volume_values_fine[residual_datapoints[j]]) - helmholtz_energy[j] for j in range(len(residual_datapoints)) ] # For each one of these volumes, compute the residual.
			else:
				residuals = [ f_eos(volume_values_fine[residual_datapoints[j]],E_0_fitted,V_0_fitted,B_0_fitted,B_0_prime_fitted) - helmholtz_energy[j] for j in range(len(residual_datapoints)) ] # For each one of these volumes, compute the residual.
			f_error = np.std(residuals)
			print('Residual fitting error for %s (eV): F_std = %f' % (structure_name,f_error))

		# Error in F & G (can only do this if the DFT enthalpies are supplied)
		if (give_errors and enthalpies):
			G_residual_datapoints = [ np.argmin ( np.abs(np.array(pressure_values_gpa_fine) - DFT_pressures[j] ) ) for j in range (len(DFT_pressures)) ]
			G_residuals = [ gibbs_free_energy[G_residual_datapoints[j]] - enthalpies[j]  for j in range(len(G_residual_datapoints)) ]
			G_std_error = np.std(G_residuals)
			print('Residual fitting error for %s (eV): F_std = %f, G_std = %f' % (structure_name,f_error,G_std_error))
		################################################################################################################
		#If the user requests a vertical line to be plotted at some pressure
		for tag in tags:
			if '#vline' in tag:
				vline = float(tag.split("=")[-1])
				ax.axvline(x=vline,linestyle='dashed',color='k')
			if '#colour' in tag:
				colour = str(tag.split("=")[-1])
				colour = colour.replace('\n','')
		################################################################################################################
		# Plotting F Vs V - Uncomment as needed
		if mode == 'plot_F':
			print(volume_values)
			print(helmholtz_energy)
			ax.plot(volume_values,helmholtz_energy,color='%s' % (colour), linestyle="none", marker="+")
			ax.plot(volume_values_fine,helmholtz_values_fitted,label=r'$\mathrm{%s}$' % (structure_name))
			plt.xlabel(r'Volume per atom (\AA$^3$)', fontsize=20, usetex=True)
			plt.ylabel(r'Helmholtz Free Energy (eV)', fontsize=20, usetex=True)
			if give_errors:
				ax.fill_between(volume_values_fine, np.array(helmholtz_values_fitted)+f_error, np.array(helmholtz_values_fitted)-f_error, alpha=0.15)
        ################################################################################################################
        # Plotting G Vs p - Uncomment as needed
		elif mode == 'plot_G':
			ax.plot(pressure_values_gpa_fine,gibbs_free_energy,label=r'$\mathrm{%s}$' % (structure_name))
			if (enthalpies and not no_enthalpy_scatter):
				ax.scatter(DFT_pressures,enthalpies,color='%s' % (colour))
			plt.xlabel(r'Pressure (%s)' % (pressure_units), fontsize=20, usetex=True)
			plt.ylabel(r'Gibbs Free Energy (eV)', fontsize=20, usetex=True)
			if give_errors:
				try:
					ax.fill_between(pressure_values_gpa_fine, gibbs_free_energy+G_std_error, gibbs_free_energy-G_std_error, alpha=0.15) # Use stds as error
				except: # This exception is triggered when there are no enthalpies in the file, and thus 'G_std_error' does not even exist
					pass
		################################################################################################################
	    # Plotting G Vs p, normalised to the reference structure - Uncomment as needed
		elif ( (mode == 'plot_G_normalise') or (mode == 'plot_pressures_normalise') ):
			if structure_name == normalising_structure: # Get the normalising structure - By construction, this should be the first structure in the list
				pressure_norm_min=pressure_values_gpa_fine[0]
				pressure_norm_max=pressure_values_gpa_fine[-1]
				pressure_grid=np.linspace(pressure_norm_min,pressure_norm_max,num_fine_points) # All data is projected onto this common pressure grid, which runs between pressure_norm_min and pressure_norm_max
				interpolated_gibbs=griddata(pressure_values_gpa_fine,gibbs_free_energy,pressure_grid)
				gibbs_energy_reference=interpolated_gibbs
			else:
				interpolated_gibbs=griddata(pressure_values_gpa_fine,gibbs_free_energy,pressure_grid)
		################################################################################################################
		if mode == 'plot_G_normalise':
			# Normalise the energy
			normalised_energy = interpolated_gibbs - gibbs_energy_reference
			gibbs_energy_reference_truncated = gibbs_energy_reference
			normalised_energy_truncated = normalised_energy
			pressure_truncated = pressure_grid
			################################################################################################################
			# Normalise the DFT pressure & enthalpy scatter points
			if (filter_outliers and enthalpies):
				press_enth_zipped=zip(DFT_pressures,enthalpies) # zip
				press_enth_zipped=[ x for x in press_enth_zipped if (x[0] > min(pressure_truncated)) and (x[0] < max(pressure_truncated)) ] # filter out datapoints that lie outside pressure_truncated
				DFT_pressures, enthalpies = zip(*press_enth_zipped) # unzip
			DFT_pressures_nearest_idx = [ np.argmin ( abs(pressure_truncated - pressure) ) for pressure in DFT_pressures  ] # finds the closest fine pressure datapoint to the individual scatter DFT pressure datapoint
			if (enthalpies and not no_enthalpy_scatter):
				scatter_enthalpy_normalised = [ (enthalpies[idx]-gibbs_energy_reference_truncated[i]) for idx,i in enumerate(DFT_pressures_nearest_idx) ]
			################################################################################################################
			# Tags
			trimlow,trimhigh=None,None
			low_shade_pressure,high_shade_pressure=None,None
			for tag in tags:
				if '#trimlow' in tag:
					trimlow=float(tag.split("=")[-1])
				if '#trimhigh' in tag:
					trimhigh=float(tag.split("=")[-1])
				if (trimlow or trimhigh):
					normalised_energy_truncated,pressure_truncated = trim_data(normalised_energy_truncated,pressure_truncated,trimlow,trimhigh)
					if (enthalpies and not no_enthalpy_scatter):
						scatter_enthalpy_normalised,DFT_pressures = trim_data(scatter_enthalpy_normalised,DFT_pressures,trimlow,trimhigh)
				if '#shadelow' in tag:
					low_shade_pressure=float(tag.split("=")[-1])
					if tpa and (low_shade_pressure>500):
						print('\'-tpa\' flag specified, so using pressure units of TPa. But #shadelow seems to be in GPa... Aborting.')
						exit()
				if '#shadehigh' in tag:
					high_shade_pressure=float(tag.split("=")[-1])
					if tpa and (high_shade_pressure>500):
						print('\'-tpa\' flag specified, so using pressure units of TPa. But #shadehigh seems to be in GPa... Aborting.')
						exit()
			################################################################################################################
			# Colour shading for lowest-enthalpy structure
			midpoint=None
			if shade_stability:
				if low_shade_pressure and high_shade_pressure:
					ax.axvspan(low_shade_pressure, high_shade_pressure, alpha=0.125, color=colour)
					midpoint = low_shade_pressure + (high_shade_pressure-low_shade_pressure) / 2
					low_shade_pressure,high_shade_pressure=None,None
				elif low_shade_pressure:
					ax.axvspan(low_shade_pressure, max(pressure_truncated), alpha=0.125, color=colour)
					midpoint = low_shade_pressure + (max(pressure_truncated)-low_shade_pressure) / 2
					low_shade_pressure=None
				elif high_shade_pressure:
					ax.axvspan(min(pressure_truncated), high_shade_pressure, alpha=0.125, color=colour)
					midpoint = min(pressure_truncated) + (high_shade_pressure-min(pressure_truncated)) / 2
					high_shade_pressure=None
			# SPACEGROUP LABELLING
			if (midpoint and label_structures):
				plt.text(midpoint, 0, r'$\mathrm{%s}$' % (structure_name),fontsize=10,horizontalalignment="center", color='k',
					bbox=dict(facecolor='white', edgecolor=colour, boxstyle='round', alpha=0.75))
			################################################################################################################
			# 'NRM' is a special reserved structure name - If a structure
			# is called 'NRM', it will not have its data plotted 
			if (structure_name != 'NRM'):
				marker = marker_tray[ count%len(marker_tray) ]
				if (enthalpies and not no_enthalpy_scatter):
					ax.scatter(DFT_pressures,scatter_enthalpy_normalised,marker=marker,s=18,color=colour)
				ax.plot(pressure_truncated,normalised_energy_truncated,linewidth=1.25,label=r'$\mathrm{%s}$' % (structure_name),color=colour,marker=marker,markevery=[])
			closest_energy_idx = np.nanargmin([x for x in normalised_energy_truncated])
			closest_energy = normalised_energy_truncated[closest_energy_idx]
			pressure_at_closest_energy = pressure_truncated[closest_energy_idx]
			furthest_energy_idx = np.nanargmax([x for x in normalised_energy_truncated])
			furthest_energy = normalised_energy_truncated[closest_energy_idx]
			pressure_at_furthest_energy = pressure_truncated[closest_energy_idx]
			if (give_minima and structure_name != normalising_structure):
				print('Smallest ΔG for %s was %f eV at %f GPa' % (structure_name,closest_energy,pressure_at_closest_energy))
			if (give_maxima and structure_name != normalising_structure):
				print('Largest ΔG for %s was %f eV at %f GPa' % (structure_name,furthest_energy,pressure_at_furthest_energy))
			if (functional is not None): # Then user requested the functional used to be labelled.
				plt.text(0.5, 0.95, functional,fontsize=15,horizontalalignment="center",transform=plt.gca().transAxes, color='blue',bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round',fc='gray', alpha=0.05))
			plt.xlabel(r'Pressure (%s)' % (pressure_units), fontsize=19, usetex=True)
			#plt.ylabel(r'Gibbs Free Energy (eV)', fontsize=19, usetex=True)
			plt.ylabel('Enthalpy (eV/atom)', fontsize=19, usetex=True)
			if give_errors:
				try:
					ax.fill_between(pressure_truncated, normalised_energy_truncated+G_std_error, normalised_energy_truncated-G_std_error, alpha=0.15)
				except: # This exception is triggered when there are no enthalpies in the file, and thus 'G_std_error' does not even exist
					pass
		################################################################################################################
		# Plotting V vs p
		elif (mode == 'plot_pressures_normalise' or mode == 'plot_pressures'):
			volume_values.reverse() # Else volumes are ordered backwards
			if use_derived_volume: # derive volume from gradient of Gibbs curve
				if (count == 0):
					print('Volumes are being derived from the derivative of G...')
				volumes_fine=np.gradient(gibbs_free_energy,pressure_values_gpa_fine) * volume_conversion_factor # V = dG/dp, with conversion to Ang^3
			else: # fit DFT volumes & pressures to a cubic spline (default)
				if count == 0: print('Volumes are being fitted to a cubic spline using the raw data...')
				V_CS=interpolator(DFT_pressures,volume_values,extrapolate=True) # no conversion factor needed - raw volumes already supplied in Ang^3
				volumes_fine=V_CS(pressure_values_gpa_fine)

			if mode == 'plot_pressures':
				# Calculate residuals...
				min_idxs = [np.argmin ( np.abs( np.array(pressure_values_gpa_fine) - p ) ) for p in DFT_pressures]
				V_residuals = [volumes_fine[idx] - volume_values[i] for i,idx in enumerate(min_idxs)]
				trimlow,trimhigh=None,None
				for tag in tags:
					if '#trimlow' in tag:
						trimlow=float(tag.split("=")[-1])
					if '#trimhigh' in tag:
						trimhigh=float(tag.split("=")[-1])
					if trimlow or trimhigh:
						volumes_fine,pressure_values_gpa_fine = trim_data(volumes_fine,pressure_values_gpa_fine,trimlow,trimhigh)
						volume_values,DFT_pressures = trim_data(volume_values,DFT_pressures,trimlow,trimhigh)
				ax.plot(pressure_values_gpa_fine,volumes_fine,label=r'$\mathrm{%s}$' % (structure_name),linewidth=2)
				ax.scatter(DFT_pressures,volume_values,marker='x')
				plt.xlabel(r'Pressure (%s)' % (pressure_units), fontsize=19, usetex=True)
				plt.ylabel(r'Volume per atom (\AA$^3$)', fontsize=19, usetex=True)

			else:
				if structure_name == normalising_structure: # Get the normalising structure - By construction, this should be the first structure in the list
					pressure_norm_min=pressure_values_gpa_fine[0]
					pressure_norm_max=pressure_values_gpa_fine[-1]
					pressure_grid=np.linspace(pressure_norm_min,pressure_norm_max,num_fine_points) # All data is projected onto this common pressure grid, which runs between 'pressure_norm_min' and 'pressure_norm_max'
					interpolated_volumes=griddata(pressure_values_gpa_fine,volumes_fine,pressure_grid)
					volume_reference=interpolated_volumes
				else:
					interpolated_volumes=griddata(pressure_values_gpa_fine,volumes_fine,pressure_grid)
				# Normalise the fine grid volumes
				normalised_volumes = interpolated_volumes-volume_reference
				plotting_pressure_min_index = np.argmin ( np.abs( pressure_grid - pressure_values_gpa_fine[0] ) )
				plotting_pressure_max_index = np.argmin ( np.abs( pressure_grid - pressure_values_gpa_fine[-1] ) )
				normalised_volumes_truncated = normalised_volumes[plotting_pressure_min_index:plotting_pressure_max_index]
				volume_reference_truncated = volume_reference[plotting_pressure_min_index:plotting_pressure_max_index]
				pressure_truncated = pressure_grid[plotting_pressure_min_index:plotting_pressure_max_index]
				# Normalise the DFT scatter raw volumes
				DFT_pressures_nearest_idx = [ np.argmin ( abs(pressure_grid - pressure) ) for pressure in DFT_pressures  ] # Finds the closest fine pressure datapoint to the individual scatter DFT pressure datapoint
				scatter_volumes_normalised = [ (volume_values[idx]-volume_reference[i]) for idx,i in enumerate(DFT_pressures_nearest_idx) ]
				###########################
				trimlow,trimhigh=None,None
				for tag in tags:
					if '#trimlow' in tag:
						trimlow=float(tag.split("=")[-1])
					if '#trimhigh' in tag:
						trimhigh=float(tag.split("=")[-1])
					if trimlow or trimhigh:
						normalised_volumes_truncated,pressure_truncated = trim_data(normalised_volumes_truncated,pressure_truncated,trimlow,trimhigh)
						scatter_volumes_normalised,DFT_pressures = trim_data(scatter_volumes_normalised,DFT_pressures,trimlow,trimhigh)
				ax.plot(pressure_truncated,normalised_volumes_truncated,label=r'$\mathrm{%s}$' % (structure_name),linewidth=2)
				ax.scatter(DFT_pressures,scatter_volumes_normalised)
				plt.xlabel(r'Pressure (%s)' % (pressure_units), fontsize=19, usetex=True)
				plt.ylabel(r'Volume per atom (\AA$^3$)', fontsize=19, usetex=True)
		################################################################################################################
		count+=1

end = time.time()
total_script_time = end - start
print('Total Script Time: %f s' % (total_script_time))

ax.legend(loc=legend_loc,ncol=legend_ncols,fontsize=12)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
if (xlim and ylim):
	plt.xlim(xlim[0],xlim[1])
	plt.ylim(ylim[0],ylim[1])
elif (xlim):
	plt.xlim(xlim[0],xlim[1])
elif (ylim):
	plt.ylim(ylim[0],ylim[1])
plt.minorticks_on()
plt.tight_layout()
plt.show()

