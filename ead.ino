// If your target is limited in memory remove this macro to save 10K RAM
//#define EIDSP_QUANTIZE_FILTERBANK   0

/* Includes ---------------------------------------------------------------- */
#include <EdgeyMonkey-project-1_inferencing.h>
#include <PDM.h>
#include <Wire.h>
#include <SeeedOLED.h>

const pin_size_t motorPin = D12;
const pin_size_t buttonPin = D8;
const unsigned char X = 3;
const unsigned char Y = 5;
int buttonState = 0;
bool alarm = false;

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
String Label;
float confidence = 0.98;

bool DEBUG = false;

// Setup Function

void setup()
{
    // put your setup code here, to run once:
    if (DEBUG) {
      Serial.begin(9600);               // initialize serial communication at 9600 bits per second:
      while (!Serial);
      Serial.println("Edge Impulse Inferencing Demo");
    }


    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) 
    {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }
    setupScreen();
    setupGpio();
    bootUp();
}

void bootUp() {
    SeeedOled.putString("Boot Up");        //Print the String
    digitalWrite(motorPin, HIGH);          // Turn motor on
    delay(5000);    // wait for 5 seconds
    SeeedOled.clearDisplay();          //clear the screen and set start position to top left corner
    SeeedOled.setNormalDisplay();      //Set display to normal mode (i.e non-inverse mode)
    SeeedOled.setPageMode();           //Set addressing mode to Page Mode
    SeeedOled.setTextXY(Y, X);         //Set the cursor to Xth Page, Yth Column
    digitalWrite(motorPin, LOW);          // Turn motor off
}

void setupScreen() {
    Wire.begin();
    SeeedOled.init();  //initialze SEEED OLED display

    SeeedOled.clearDisplay();          //clear the screen and set start position to top left corner
    SeeedOled.setNormalDisplay();      //Set display to normal mode (i.e non-inverse mode)
    SeeedOled.setPageMode();           //Set addressing mode to Page Mode
    SeeedOled.setTextXY(Y, X);         //Set the cursor to Xth Page, Yth Column
}

void setupGpio() {
  // initialize the motor pin as an output:
  pinMode(motorPin, OUTPUT);
  // initialize the pushbutton pin as an input:
  pinMode(buttonPin, INPUT);
}

// This will always loop
void loop()
{   
    if (alarm == true) {

      // read the state of the pushbutton value:
      buttonState = digitalRead(buttonPin);

      // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
      if (buttonState == HIGH) {
        digitalWrite(motorPin, LOW);
        SeeedOled.clearDisplay();          //clear the screen and set start position to top left corner
        SeeedOled.setNormalDisplay();      //Set display to normal mode (i.e non-inverse mode)
        SeeedOled.setPageMode();           //Set addressing mode to Page Mode
        SeeedOled.setTextXY(Y, X);         //Set the cursor to Xth Page, Yth Column
        alarm = false;
      } 
    } else {

      ei_printf("Starting inferencing in 2 seconds...\n");

      delay(20);

      ei_printf("Recording...\n");

      bool m = microphone_inference_record();
      if (!m) {
          ei_printf("ERR: Failed to record audio...\n");
          return;
      }

      ei_printf("Recording done\n");

      signal_t signal;
      signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
      signal.get_data = &microphone_audio_signal_get_data;
      ei_impulse_result_t result = { 0 };

      EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
      if (r != EI_IMPULSE_OK) {
          ei_printf("ERR: Failed to run classifier (%d)\n", r);
          return;
      }

      // print the predictions
      ei_printf("Predictions ");
      ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
          result.timing.dsp, result.timing.classification, result.timing.anomaly);
      ei_printf(": \n");
      for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) 
      {
          ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
          if(result.classification[ix].value > confidence )
                  {
                      Label = String(result.classification[ix].label);
                      if (DEBUG) {
                        Serial.println(Label);
                      }
                      if (Label == "Fire Alarm")
                      {
                        alarm = true;
                        SeeedOled.putString("Fire Alarm!"); 
                        digitalWrite(motorPin, HIGH);

                        if (DEBUG) {
                          Serial.println("Fire Detected");
                        }
                      }
                      else if (Label == "Background")
                      {
                        if (DEBUG) {
                          Serial.println("Background Noise");
                        }
                      }
                    }
      }
    }
    
    
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
}


static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if(inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize(4096);

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();

        return false;
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    return true;
}

static bool microphone_inference_record(void)
{
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while(inference.buf_ready == 0) {
        delay(10);
    }

    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);

    return 0;
}


static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
