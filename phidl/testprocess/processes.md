[//]: # (wsi_deposition_01)

# WSi Deposition for SNSPDs

- PRECLEAN: O2 plasma at 60W, 50sccm O2 for 3min before loading wafer into AJA
- SPUTTER: 4.2nm WSi (cosputtered W + Si)
    -  Set 1.2mTorr; 10sccm Ar; 100W on W gun, 180W on Si gun
    -  Pre-soak guns for 2min; record voltage and current at 1min;
    -  Dep for 28 s
        - Voltage: ________________ V
        - Current: ________________ A
    -  Close shutter
- SPUTTER: 2nm aSi
    -  Set 10mTorr; 15sccm Ar
    -  Ramp Si gun 180W -> 475W
    -  Dep for 15 s
        - Voltage: ________________ V
        - Current: ________________ A
- MEASURE: Four point resistivity
    - Measured resistance R: ________________  Ohms
    - Sheet resistance Rs: For a large wafer Rs = R pi/ln(2) ~= 4.5R = ________________  Ohms/sq


[//]: # (gold_liftoff_01)

# Patterned gold ground layer liftoff process

- PATTERN on stepper
    + Spin: P20 at 4000rpm, SPR 660 at 3000rpm
    + Bake: hotplate with vacuum contact at 95C° for 1min
    + Expose: ASML 5500 100D; 180mJ/cm2
    + Post-exposure bake: 110C for 1min
    + Develop: 30s+30s double puddle
    + Inspect with microscope
- EVAPORATE: 5nm Ti + 50nm Au + 5nm Ti
    + Settings go here
    + Other stuff blah blah
- LIFTOFF in NMP
    + Soak in acetone briefly; liftoff is easy with this film.
    + Sonicate in dirty beaker until film is visibly removed from wafer. Then do 5 more minutes in dirty, 5 in clean.
    + Spray with acetone, then with IPA, then soak in IPA beaker. Go from IPA beaker to SRD.

----------
[//]: # (gold_liftoff_02)

# Patterned gold ground layer liftoff process

- Here's a new step not used before
    + Post-exposure bake: 110C for 1min
    + Develop: 30s+30s double puddle
    + Inspect with microscope
- EVAPORATE: 5nm Ti + 50nm Au + 5nm Ti
- LIFTOFF in NMP
