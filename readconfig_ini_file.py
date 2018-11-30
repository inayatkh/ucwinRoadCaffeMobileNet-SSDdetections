#!/usr/bin/env python3


import sys
import configparser
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A program that Bash can call to parse an .ini file")
    parser.add_argument("inifile", help="name of the .ini file")
    parser.add_argument("section", help="name of the section in the .ini file")
    parser.add_argument("itemname", help="name of the desired value")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.inifile)
    print(config.get(args.section, args.itemname))