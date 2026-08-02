'../Dielectric'                                 ! Path for the dielectric constant
'p' 						! Polarization of the incident beam (s;p)
1.5						! Photon energy range min (eV)
5.						! Photon energy range max (eV)
0.785398163					! Incident angles (rad) 
island						! Geometry used (island,Yamaguchi,coated)
ag						! Island material (see database)
mgo						! Substrate material (see database)
tio2t						! Coating material (see database)
finite_size	   				! Mean free path of electrons (finite_size,s-only,tau,manual,A-parameter)
F       	 				! Surface effects	(F,T)
2.5	  					! Island radius parallel (nm) — prolate: R_par < R_per
5.						! Island radius perpendicular (nm)
1.						! Thickness of the layer or coating (nm)
0.001						! Truncation parameter
MFT						! The type of network for interactions (square,hexagonal,MFT,RPT)
0.236						! Coverage 
30.						! Lattice constant
dipole       					! Island-island interaction (none,dipole,quadrupole)
constitutive					! Type of formulae for DR (constitutive,constitutive_all,invariants,aspnes)
DR						! Nature of output (DR,R,T,DT,A)
F						! Ouput for the multipoles coefficients (F,T)
F						! Output for the polarizability (F,T)	
100						! # points in energy
10						! # multipoles
none						! Potential (none,Pot)
3						! Energy for potential or boundary condition (eV)
ProlateTest					! Outfilename
quadpack	                        	! Integration method (quadpack,gauleg)
F						! Comparison with experiment (F,T)
agmgo						! Experiment file 
