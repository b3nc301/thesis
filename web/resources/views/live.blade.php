<head>
    <title>Detektor beállításai</title>

 <link href="https://vjs.zencdn.net/7.2.3/video-js.css" rel="stylesheet">


</head>
    <x-app-layout>

        <x-slot name="header">
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">
                {{ __('Detektor beállírásai') }}
            </h2>
        </x-slot>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                    <div class="p-6 bg-white border-b border-gray-200">
                        <div id="liveAlert"></div>
                        <video
                        id="live-stream"
                        controls
			            class="video-js vjs-default-skin mb-5"
                        width="1280"
                        height="720">
                        <source src="/lives/stream.m3u8" type="application/x-mpegURL" />
                      </video>



                    <div class="liveButtons">
                      <div>Detektor beállított IP címe: <b>{{env('DETECTOR_IP')}}</b></div>
                      <div class="liveControl">
                          <button type="button" class="btn btn-success" onclick=startDetector()> Detektor indítása</button>
                          <button type="button" class="btn btn-danger" onclick=stopDetector()> Detektor leállítása</button>
                          <button type="button" class="btn btn-primary" onclick=getData()> Adatok lekérése</button>
                          <button type="button" class="btn btn-primary" onclick=changeData()> Adatok módosítása</button>
                        </div>
                      <div>Videóanyag forrása:</div> <input type="text" id="src">
                      <div>Bizonyossági küszöb:</div> <input type="number" id="conf">
                      <div>Minimum szükséges távolság emberek között(cm):</div> <input type="number" id="min">
                    </div>
                    </div>
                </div>
            </div>
        </div>

    </x-app-layout>
    <script>
        var ip = "<?php echo(env('DETECTOR_IP')) ?>";
        var port = "<?php echo(env('DETECTOR_PORT')) ?>"
    </script>
    <script src="/js/socket.js"></script>


<!-- JS code -->
<!-- If you'd like to support IE8 (for Video.js versions prior to v7) -->
<script src="https://vjs.zencdn.net/ie8/ie8-version/videojs-ie8.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/videojs-contrib-hls/5.14.1/videojs-contrib-hls.js"></script>
<script src="https://vjs.zencdn.net/7.2.3/video.js"></script>

<script>
var player = videojs('live-stream');
player.play();
</script>
