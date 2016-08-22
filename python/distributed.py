#!/usr/bin/env python

from __future__ import print_function
import zmq
import sys
import time
import argparse

def server(argv=None):
    print("Running server...")
    #parser.add_argument("directory",
    #                    metavar="DIRECTORY",
    #                    nargs="?",
    #                    help="specify directory of factor graph files",
    #                    default=".",
    #                    type=str)
    #parser.add_argument("-m", "--meta",
    #                    metavar="META_FILE",
    #                    dest="meta",
    #                    default="graph.meta",
    #                    type=str,
    #                    help="meta file") # TODO: print default for meta, weight, variable, factor in help
    #parser.add_argument("-w", "--weight",
    #                    metavar="WEIGHTS_FILE",
    #                    dest="weight",
    #                    default="graph.weights",
    #                    type=str,
    #                    help="weight file")
    #parser.add_argument("-v", "--variable",
    #                    metavar="VARIABLES_FILE",
    #                    dest="variable",
    #                    default="graph.variables",
    #                    type=str,
    #                    help="variable file")
    #parser.add_argument("-f", "--factor",
    #                    metavar="FACTORS_FILE",
    #                    dest="factor",
    #                    default="graph.factors",
    #                    type=str,
    #                    help="factor file")
    ## TODO: burn-in option
    ## TODO: learning options
    ## TODO: inference option
    #parser.add_argument("-l", "--learn",
    #                    metavar="NUM_LEARN_STEPS",
    #                    dest="learn",
    #                    default=0,
    #                    type=int,
    #                    help="number of learning sweeps")
    #parser.add_argument("-i", "--inference",
    #                    metavar="NUM_INFERENCE_STEPS",
    #                    dest="inference",
    #                    default=0,
    #                    type=int,
    #                    help="number of inference sweeps")
    ## TODO: sample observed variable option
    #parser.add_argument("-q", "--quiet",
    #                    #metavar="QUIET",
    #                    dest="quiet",
    #                    default=False,
    #                    action="store_true",
    #                    #type=bool,
    #                    help="quiet")
    ## TODO: verbose option (print all info)
    #parser.add_argument("--verbose",
    ##                    metavar="VERBOSE",
    #                    dest="verbose",
    #                    default=False,
    #                    action="store_true",
    ##                    type=bool,
    #                    help="verbose")
    #parser.add_argument("--version",
    #                    action='version',
    #                    version="%(prog)s 0.0",
    #                    help="print version number")
    port = "5556"
    #if len(sys.argv) > 1:
    #    port =  sys.argv[1]
    #    int(port)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    num_clients = 0

    while True:
        #  Wait for next request from client
        message = socket.recv()
        if len(message) < 1:
            print("Received message of size 0.", file=sys.stderr)
            socket.send("?") # Need to reply or this crashes
        elif message[0] == 'a': # Initial message from client
            print("Received hello.")
            socket.send("%d" % num_clients)
            num_clients += 1
        elif message[0] == 'b': # Request for factor graph
            print("Received request for factor graph.")
            # TODO
            socket.send("%d" % num_clients)
            num_clients += 1
        else:
            print("Message cannot be interpreted.", file=sys.stderr)


    return

def client(argv=None):
    print("Running client...")
    port = "5556"
    #if len(sys.argv) > 1:
    #    port =  sys.argv[1]
    #    int(port)

    #if len(sys.argv) > 2:
    #    port1 =  sys.argv[2]
    #    int(port1)

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect ("tcp://localhost:%s" % port)
    if len(sys.argv) > 2:
        socket.connect ("tcp://localhost:%s" % port1)

    socket.send ("a")
    message = socket.recv()
    client_id = int(message)
    print("Received id #%d." % client_id)

    # TODO: request factor graph if not loaded

    # Send "ready"
    socket.send("c")

    while True:
        message = socket.recv()
        if len(message) < 1:
            print("Received message of size 0.", file=sys.stderr)
            socket.send("?") # Need to reply or this crashes
        elif message[0] == 'a': # Initial message from client
            print("Received hello.")
            socket.send("%d" % num_clients)
            num_clients += 1
        elif message[0] == 'b': # Request for factor graph
            print("Received request for factor graph.")
            # TODO
            socket.send("%d" % num_clients)
            num_clients += 1
        else:
            print("Message cannot be interpreted.", file=sys.stderr)

    ##  Do 10 requests, waiting each time for a response
    #for request in range (1,10):
    #    print("Sending request ", request,"...")
    #    socket.send ("Hello")
    #    #  Get the reply.
    #    message = socket.recv()
    #    print("Received reply ", request, "[", message, "]")
    #return

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run a parameter server/workers",
        epilog="")

    parser.add_argument("program",
                        metavar="PROGRAM",
                        nargs="?",
                        help="server/client (s/c)",
                        default="s",
                        type=str)

    arg = parser.parse_args(argv)

    if arg.program.lower() == "server" or arg.program.lower() == "s":
        server(argv)
    elif arg.program.lower() == "client" or arg.program.lower() == "c":
        client(argv)
    else:
        print("Error:", arg.program, "is not a valid choice.", file=sys.stderr)



if __name__ == "__main__":
    main()

