# Creating the job and stepper files

## Loading GDS files onto Grumpy / converting with reticleset

[Detailed BMF wiki instructions here](http://doc.bqemd.nist.gov/bmf/equipment/heidelberg_dwl2k/cad_file_conversion)

- ssh into `grumpy.boulder.nist.gov` with your username
- Copy your .gds files into a folder your home directory
- The CAD conversion software for GDS is done by running the command `reticleset`
- Select layers you want (remember to invert if doing an etch!)
- Once `reticleset` is done, hit "Upload to PG" and email "Job Report" to yourself

## Using the convert computer

[Detailed BMF wiki instructions here](http://doc.bqemd.nist.gov/bmf/equipment/heidelberg_dwl2k/cad_file_conversion#convert_computer)

- The convert computer handles the actual conversion to the Heidelberg interface
- ssh into `686bmf-patgen.bw.nist.gov` with username `convert`
- Change directory to where the converted files are (your files will likely be in `~/opto/username/`)
- In the folder with the converted .tgz file, run  `pgfilesetup`
- Now the files will be available on the Heidelberg operation computer - follow the SOP printout
- Once completed, run `pgfilecleanup`

## Making job files for the stepper

[Detailed BMF wiki instructions here](http://doc.bqemd.nist.gov/bmf/equipment/asml_5500/writing_jobfiles_for_pas5500)

- ssh into `686bmf-asmlsaw.bw.nist.gov` with username `sys.8333`. No password is required.
- Run `pas_saw_control`