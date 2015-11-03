#ifndef _MESSAGE_H
#define _MESSAGE_H

enum FusionMessageType {
  REQUEST_GRAD  = 0,
  RESPONSE_GRAD = 1,
  REQUEST_ACCURACY = 2,
  RESPONSE_ACCURACY = 3
};


struct FusionMessage {

  FusionMessageType msg_type;

  int nelem;

  int batch;

  int imgids[256];

  int labels[256];

  float content[0];

  int size(){
    return sizeof(FusionMessage) + nelem*sizeof(float);
  }

};

#endif