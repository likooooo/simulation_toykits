'../Dielectric'                                 ! Path for the dielectric constant (NB : use '..' under unix)
island						! Geometry (film,2film,thin_cap,island)
0.  0.						! Film_thickness1, Film_thickness2 (thin films-nm)
5.  0.75  0.					! R (nm), tr, MPpos
dipole   hexagonal  13.2 0.45				! Interaction (none,dipole,quadrupole), Network (square, hexagonal,MFT,RPT), Lattice_const (nm), coverage
DR   p  0.785398163  constitutive 		! Out (R,DR,T,Dt,A), Polarization (s,p), Theta0 (rad), Fresnel (constitutive, constitutive_all,invariants,aspnes)
1.5 4.5 100 					! Energy_min (eV), Energy_max (eV), Nenergy
ag   mgo   ti 					! Island, Substrate, Coating (see dielectric database)
finite_size  F					! Mean_free_path (none,finite_size,tau,manual,A-parameter), Surface_effects (T,F)
F F F						! Polarizability (T,F), Normalization (T,F), Strenght (T,F)
none  2.5  incident_field 			! Bound (none,Pot,BC), BC_energy (eV), Scaling (none, incident_field, mean_value)
8						! Mpole_order
HexagonalTest 						! Outfilename
F  agmgo					! Compa (T,F), Expfilename

