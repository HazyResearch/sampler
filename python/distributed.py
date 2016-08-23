#!/usr/bin/env python

from __future__ import print_function
import zmq
import sys
import time
import argparse
import gibbs
import numpy as np

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=eval(md['dtype']))
    return A.reshape(md['shape'])

def server(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Gibbs worker",
        epilog="")

    parser.add_argument("directory",
                        metavar="DIRECTORY",
                        nargs="?",
                        help="specify directory of factor graph files",
                        default="",
                        type=str)
    parser.add_argument("-p", "--port",
                        metavar="PORT",
                        help="port",
                        default=5556,
                        type=int)
    parser.add_argument("-m", "--meta",
                        metavar="META_FILE",
                        dest="meta",
                        default="graph.meta",
                        type=str,
                        help="meta file") # TODO: print default for meta, weight, variable, factor in help
    parser.add_argument("-w", "--weight",
                        metavar="WEIGHTS_FILE",
                        dest="weight",
                        default="graph.weights",
                        type=str,
                        help="weight file")
    parser.add_argument("-v", "--variable",
                        metavar="VARIABLES_FILE",
                        dest="variable",
                        default="graph.variables",
                        type=str,
                        help="variable file")
    parser.add_argument("-f", "--factor",
                        metavar="FACTORS_FILE",
                        dest="factor",
                        default="graph.factors",
                        type=str,
                        help="factor file")
    parser.add_argument("-b", "--burn",
                        metavar="NUM_BURN_STEPS",
                        dest="burn",
                        default=0,
                        type=int,
                        help="number of learning sweeps")
    parser.add_argument("-l", "--learn",
                        metavar="NUM_LEARN_STEPS",
                        dest="learn",
                        default=0,
                        type=int,
                        help="number of learning sweeps")
    parser.add_argument("-e", "--epoch",
                        metavar="NUM_LEARNING_EPOCHS",
                        dest="epoch",
                        default=0,
                        type=int,
                        help="number of learning epochs")
    parser.add_argument("-i", "--inference",
                        metavar="NUM_INFERENCE_STEPS",
                        dest="inference",
                        default=0,
                        type=int,
                        help="number of inference sweeps")
    # TODO: sample observed variable option
    parser.add_argument("-q", "--quiet",
                        #metavar="QUIET",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        #type=bool,
                        help="quiet")
    # TODO: verbose option (print all info)
    parser.add_argument("--verbose",
    #                    metavar="VERBOSE",
                        dest="verbose",
                        default=False,
                        action="store_true",
    #                    type=bool,
                        help="verbose")

    print("Running server...")

    arg = parser.parse_args(argv[1:])

    print(arg.directory)

    if arg.directory == "":
        fg = None
    else:
        var_copies = 1
        weight_copies = 1
        (meta, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate) = gibbs.load(arg.directory, arg.meta, arg.weight, arg.variable, arg.factor, not arg.quiet, not arg.verbose)
        fg = gibbs.FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate, var_copies, weight_copies)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % arg.port)

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
            # TODO: check that fg != None
            socket.send("%d" % num_clients)
            num_clients += 1
        elif message[0] == 'c': # Client ready
            print("Received ready.")
            socket.send("d%d" % arg.burn) # could skip this if arg.burn == 0
        elif message[0] == 'e' or message[0] == 'g': # Client done with burn/learning
            if message[0] == 'e': # Done burning
                epochs = 0
            else: # Done learning
                epochs = socket.recv_json()
                fg.wv += recv_array(socket)
                pass

            if epochs < arg.epoch:
                socket.send("f", zmq.SNDMORE) # Do learning
                socket.send_json(arg.learn, zmq.SNDMORE)
                socket.send_json(0.001, zmq.SNDMORE) # TODO
                send_array(socket, fg.wv)
            else:
                socket.send("h%d" % arg.inference) # Do inference
        elif message[0] == 'i': # Client done with inference
            data = recv_array(socket)
            # TODO: handle count
            socket.send("j")
        else:
            print("Message cannot be interpreted.", file=sys.stderr)
            socket.send("j")

    return

def client(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Gibbs worker",
        epilog="")

    parser.add_argument("directory",
                        metavar="DIRECTORY",
                        nargs="?",
                        help="specify directory of factor graph files",
                        default="",
                        type=str)
    parser.add_argument("-p", "--port",
                        metavar="PORT",
                        help="port",
                        default=5556,
                        type=int)
    parser.add_argument("-m", "--meta",
                        metavar="META_FILE",
                        dest="meta",
                        default="graph.meta",
                        type=str,
                        help="meta file") # TODO: print default for meta, weight, variable, factor in help
    parser.add_argument("-w", "--weight",
                        metavar="WEIGHTS_FILE",
                        dest="weight",
                        default="graph.weights",
                        type=str,
                        help="weight file")
    parser.add_argument("-v", "--variable",
                        metavar="VARIABLES_FILE",
                        dest="variable",
                        default="graph.variables",
                        type=str,
                        help="variable file")
    parser.add_argument("-f", "--factor",
                        metavar="FACTORS_FILE",
                        dest="factor",
                        default="graph.factors",
                        type=str,
                        help="factor file")
    parser.add_argument("-q", "--quiet",
                        #metavar="QUIET",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        #type=bool,
                        help="quiet")
    parser.add_argument("--verbose",
    #                    metavar="VERBOSE",
                        dest="verbose",
                        default=False,
                        action="store_true",
    #                    type=bool,
                        help="verbose")

    print(argv)
    arg = parser.parse_args(argv[1:])

    print("Running client...")
    print(arg.directory)

    if arg.directory == "":
        fg = None
    else:
        var_copies = 1
        weight_copies = 1
        (meta, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate) = gibbs.load(arg.directory, arg.meta, arg.weight, arg.variable, arg.factor, not arg.quiet, not arg.verbose)
        fg = gibbs.FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate, var_copies, weight_copies)

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect ("tcp://localhost:%s" % arg.port)

    # hello message
    socket.send ("a")
    message = socket.recv()
    client_id = int(message)
    print("Received id #%d." % client_id)

    # request factor graph if not loaded
    if fg == None:
        socket.send("b")
        message = socket.recv()
        # TODO: generate factor graph

    # Send "ready"
    socket.send("c")

    learning_epochs = 0
    while True:
        message = socket.recv()
        if len(message) < 1:
            print("Received message of size 0.", file=sys.stderr)
            socket.send("?") # Need to reply or this crashes
        elif message[0] == 'd': # request for burn-in
            print("Received request for burn-in.")
            burn = int(message[1:])
            print("Burning", burn, "sweeps.")
            fg.gibbs(burn, 0, 0)
            socket.send("e")
        elif message[0] == 'f': # Request for learning
            print("Received request for learning.")
            sweeps = socket.recv_json()
            step = socket.recv_json()
            print(sweeps)
            print(step)
            fg.wv = recv_array(socket)
            wv = fg.wv

            fg.learn(sweeps, step, 0, 0)

            dw = fg.wv - wv
            socket.send("g", zmq.SNDMORE)
            learning_epochs += 1
            socket.send_json(learning_epochs, zmq.SNDMORE)
            send_array(socket, dw)
        elif message[0] == 'h': # Request for inference
            print("Received request for inference.")
            inference = int(message[1:])
            print("Inference:", inference, "sweeps.")
            fg.clear()
            fg.gibbs(inference, 0, 0)
            socket.send("i", zmq.SNDMORE)
            send_array(socket, fg.count)
        elif message[0] == 'j': # Exit
            print("Exit")
            break
        else:
            print("Message cannot be interpreted.", file=sys.stderr)
            break

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print("Usage: ./distributed.py [server/client]", file=sys.stderr)
    elif argv[0].lower() == "server" or argv[0].lower() == "s":
        server(argv)
    elif argv[0].lower() == "client" or argv[0].lower() == "c":
        client(argv)
    else:
        print("Error:", argv[0], "is not a valid choice.", file=sys.stderr)

if __name__ == "__main__":
    main()

