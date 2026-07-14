1/04/2026 SRO

Issues with cvs files:

Results file contains data not in beta test.  -- FIXED - using cut csv file
This means the rows don't always have the same values.


Images:
Images with ringed insects are only from beta test so doesnt match fully with the cvs files.

Colours of rings:
	    'Thrip': "blue",
            'Pirate Bug': "green",
            'Other Insect' : 'red',
            'Possible Insect':'purple'


Issues to account for:
       code not optimal what so ever. Its a hatch job, could be 'prettified' using pnadas.
       not currently taking in to account any overlap in images.
       
       

How to run the codes:

Taking the raw Zooniverse output, in this case rhs-wisley-bug-watch-classifications_cut.csv run the following command:

1) python Process_Zooniverse_Data.py rhs-wisley-bug-watch-classifications_cut.csv

This will output two files, with different levels of processing:

Zooniverse_results.csv - similar to original file, but more user friendly
Final_insect_numbers.csv - data corresponding to number of insects per subimage

To get some statistics and output subimages with insects marked up run in a directory you desire somethi
2) python PATH_TO/Plot_Insects.py /PATH_TO/Final_insect_numbers.csv -makeimages no

If you do not include " -makeimages no " it will make a folder called ResultImages/ which contains a copy of each of the subimages with an insect identified with the insect(s) marked with a circle.

To get some basic statistics on number of classifications and participants run:
3) python PATH_TO/basic_project_stats.py /PATH_TO/rhs-wisley-bug-watch-classifications_cut.csv


rhs-wisley-bug-watch-classifications_cut.csv - is a cut version of the data downloaded straight from Zooniverse. It has been cut to remove entried before the beta test started.
