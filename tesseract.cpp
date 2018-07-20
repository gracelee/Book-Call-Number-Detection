#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>



//using namespace cv;
//using namespace std;

int main(int argc, char** argv) {
        // [1]
        tesseract::TessBaseAPI *myOCR =
                new tesseract::TessBaseAPI();

        char filename[80];
        std::ifstream file("list.txt");
        std::string str;
        myOCR->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345789");

        // [3]
        if (myOCR->Init(NULL, "eng")) {
          fprintf(stderr, "Could not initialize tesseract.\n");
          exit(1);
        }


        // [4]
        for (size_t i=0; i<24; i++) {
          sprintf(filename,"/home/cosc/student/dil15/Desktop/Project/letters_%d.tiff",i);
          //sprintf(filename,argv[1]);
          //cout << filename << endl;
        Pix *pix = pixRead(filename);
        myOCR->SetImage(pix);

        char* outText = myOCR->GetUTF8Text();
        printf("OCR output:\n\n");
        printf(outText);

        myOCR->Clear();
        myOCR->End();
        delete [] outText;
        pixDestroy(&pix);
        if (myOCR->Init(NULL, "eng")) {
          fprintf(stderr, "Could not initialize tesseract.\n");
          exit(1);
        }

      }
      return 0;


}
