/*
 * mic_test.ino - INMP441 sanity check on a classic ESP32 (legacy I2S driver).
 *
 * Wiring:  INMP441 VDD->3V3   GND->GND   SCK->GPIO26   WS->GPIO25   SD->GPIO34   L/R->GND
 * Open Tools > Serial Plotter at 115200. Clap or speak: the trace must jump.
 *   flat line near 0      -> SD/SCK/WS wiring wrong, or VDD not on 3V3
 *   huge constant offset  -> L/R floating: tie it to GND
 */
#include <Arduino.h>
#include <driver/i2s.h>
#include <string.h>

static const i2s_port_t PORT = I2S_NUM_0;
static int32_t buf[256];

void setup() {
  Serial.begin(115200);
  i2s_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = 16000;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;      // L/R pin tied to GND = left
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 4;
  cfg.dma_buf_len = 256;
  i2s_driver_install(PORT, &cfg, 0, NULL);

  i2s_pin_config_t pins;
  memset(&pins, 0, sizeof(pins));
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = 26;
  pins.ws_io_num = 25;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = 34;
  i2s_set_pin(PORT, &pins);
}

void loop() {
  size_t got = 0;
  i2s_read(PORT, buf, sizeof(buf), &got, portMAX_DELAY);
  int n = got / 4;
  int32_t peak = 0;
  for (int i = 0; i < n; i++) {
    int32_t s = buf[i] >> 14;                          // 24-bit data sits in the top of the 32-bit slot
    if (s < 0) s = -s;
    if (s > peak) peak = s;
  }
  Serial.println(peak);
}
