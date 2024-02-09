# Call Jerman vessel enhancement code from python

import argparse
import matlab.engine as m_engine

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="Name of the file to be processed")

args = parser.parse_args()

filename = args.filename

Engine = m_engine.start_matlab()

# Engine.vess_runner(fname)
Engine.Vessel2d(filename)

