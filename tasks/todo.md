# Bookshelf Counter — Implementation Checklist

## Phase 2: Backend (YOLO-World)
- [x] Project folder structure created
- [x] `backend/requirements.txt` written
- [x] `backend/model_adapter.py` — abstract base class
- [x] `backend/nms.py` — NMS helper
- [x] `backend/adapters/yolo_world.py` — YOLO-World-S adapter
- [x] `backend/adapters/locate_anything.py` — LocateAnything stub
- [x] `backend/main.py` — FastAPI app
- [x] `backend/test_detect.sh` — curl smoke test
- [x] Backend tested: health=ok, warm inference=66ms (MPS on Apple Silicon)

## Phase 3: Android UI shell
- [ ] Gradle config scaffolded (settings, root build, app build, version catalog)
- [ ] AndroidManifest.xml with permissions + FileProvider
- [ ] Resources (file_paths.xml, strings.xml, themes.xml)
- [ ] HomeScreen.kt — camera + gallery buttons, prompt input
- [ ] MainActivity.kt — NavHost

## Phase 4: Integration + BoxOverlay
- [ ] data/model/DetectionResponse.kt
- [ ] data/DetectionApi.kt — Retrofit interface
- [ ] data/DetectionRepository.kt
- [ ] viewmodel/DetectionViewModel.kt
- [ ] ui/components/BoxOverlay.kt — Canvas + tap-to-remove
- [ ] ui/ResultScreen.kt — image + overlay + controls
- [ ] End-to-end test: emulator + `adb reverse tcp:8000 tcp:8000`

## Phase 5: LocateAnything (GPU required)
- [ ] Implement `backend/adapters/locate_anything.py`
- [ ] Test with `ADAPTER=locate_anything uvicorn main:app`
- [ ] Benchmark vs YOLO-World

## Notes
- Backend URL for emulator: `http://10.0.2.2:8000/`
- Physical device: run `adb reverse tcp:8000 tcp:8000` or change BASE_URL to machine IP
- Apple Silicon Macs: YOLO-World uses MPS automatically via Ultralytics
- LocateAnything needs ~10 GB VRAM (float16) — defer until GPU confirmed
