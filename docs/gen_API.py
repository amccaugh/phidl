import os
import sys
from operator import itemgetter
from os.path import dirname, exists, join


def write_docstring(
    main_path,
    source_path,
    fread,
    fwrite="API.rst",
    main_header="API Reference",
    sub_header="Library",
):
    """Writes a Sphinx-markdown-formatted file that generates docstrings for
    all classes and functions within the ``fname`` Python file.

    Parameters
    ----------
    main_path : str
        The path to the main package directory.
    source_path : str
        The path to the Sphinx source folder containing the .rst files.
    fname : str
        Name of the .py file to generate docstrings for (include extension).
    fwrite : str
        Name of the .rst to write docstrings to (include extension).
    main_header : str
        The top-level header for the markdown file.
    sub_header : str, optional
        Appended to the subheader associated with the ``fname`` (e.g. if
        ``fname`` = 'test.py' and ``sub_header`` = 'Library', the section
        header for ``test.py`` becomes 'Test Library').

    Notes
    -----
    If a file of the name ``fwrite`` is passed into the function, it will be
    deleted completely and rebuilt. If no file of that name exists, one will
    be created.

    Classes are placed at the top of the document, with functions following,
    and both are internally sorted alphabetically. Functions that have leading
    underscores will be ignored.
    Ex:
        def _z(*):
        class Z(*):
        def f(*):
        def a(*):
        class G(*):
        class B(*):

        will be sorted as B, G, Z, a, f.
    """
    os.chdir(main_path)
    fr = open(fread, "r")
    flines = fr.readlines()
    fr.close()
    os.chdir(source_path)
    fw = open(fwrite, "a+")

    # Creating headers
    if os.stat(fwrite).st_size == 0:
        fw.write(
            "#" * len(main_header)
            + "\n{}\n".format(main_header)
            + "#" * len(main_header)
            + "\n\n\n"
        )
    if sub_header is not None:
        sub_header = " " + sub_header
    else:
        sub_header = ""
    fread_header = fread_name = fread[:-3]
    if "_" in fread_header:
        fread_header = (
            fread_header.split("_")[0].capitalize()
            + " "
            + fread_header.split("_")[1].capitalize()
        )
    else:
        fread_header = fread_header.capitalize()
    fw.write(
        "*" * (len(fread_header) + len(sub_header))
        + "\n{}{}\n".format(fread_header, sub_header)
        + "*" * (len(fread_header) + len(sub_header))
        + "\n\n"
    )

    # Gathering all functions and classes as tuples of (name, type).
    # Single-letter functions are appended a unique string to prevent them from
    # being incorrectly sorted.
    names = []
    for line in flines:
        if line[0:5] == "class":
            cname = line[6 : line.find("(")]
            if cname[-1] == ":":
                cname = cname[:-1]
            if cname[0] != "_":
                names.append((cname, "C"))
        if line[0:3] == "def":
            fname = line[4 : line.find("(")]
            if len(fname) == 1:
                fname = fname.lower() + "GENAPIFUNC"
            if fname[0] != "_":
                names.append((fname, "F"))
    # Sorting by type first and then alphabetically
    names = sorted(names, key=itemgetter(1, 0))

    for name in names:
        # Fixing single-letter function names
        if "GENAPIFUNC" in name[0]:
            name = (name[0].replace("GENAPIFUNC", ""), "F")
            name = (name[0].upper(), "F")

        # Writing to file in Sphinx markdown format
        fw.write(name[0] + "\n")
        fw.write(("=" * len(name[0])) + "\n\n")
        if name[1] == "C":
            fw.write(".. autoclass:: phidl.{}.{}\n".format(fread_name, name[0]))
            fw.write(
                "   :members:\n"
                "   :inherited-members:\n"
                "   :show-inheritance:\n\n\n"
            )
        else:
            fw.write(".. autofunction:: phidl.{}.{}\n\n\n".format(fread_name, name[0]))
    fw.close()


# Gathering command-line arguments
source_path = os.getcwd()
args = sys.argv
if args[1][-4:] == ".rst":
    fwrite = args[1]
    args = args[2:]
    try:
        os.remove(join(source_path, fwrite))
    except Exception:
        pass
elif args[1] == "add":
    fwrite = args[2]
    args = args[3:]
elif args[1] == "build":
    os.chdir(dirname(source_path))
    os.system("sphinx-build -b html source build")
    sys.exit()
else:
    raise ValueError("First input argument must be a .rst file, " '"add", or "build".')

# Determining necessary paths
py_path = input(
    "Name of the project subfolder containing the package " "Python files: "
)
main_path = join(dirname(dirname(source_path)), py_path)
if exists(join(main_path, args[0])):
    pass
else:
    raise ValueError("Python subfolder path is not valid.")

# Executing write_docstring
for i in range(len(args)):
    if args[i][-3:] != ".py":
        raise ValueError("All input arguments past the first must be Python" " files.")
    write_docstring(
        main_path=main_path, source_path=source_path, fread=args[i], fwrite=fwrite
    )

# pip uninstall phidl
# pip install git+https://github.com/amccaugh/phidl.git@docs
os.chdir(dirname(source_path))
os.system("sphinx-build -b html source build")

# cd E:\Documents\GitHub\phidl\docs\source
# python gen_API.py API.rst geometry.py device_layout.py
