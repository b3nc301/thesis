<head>
    <title>Live video</title>

 <link href="https://vjs.zencdn.net/7.2.3/video-js.css" rel="stylesheet">


</head>
    <x-app-layout>

        <x-slot name="header">
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">
                {{ __('Video Settings') }}
            </h2>
        </x-slot>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                    <div class="p-6 bg-white border-b border-gray-200">
                        <div id="liveAlert"></div>
                        <video
                        id="my-video"
                        controls
			class="video-js vjs-default-skin mb-5"
                        width="1280"
                        height="720">
                        <source src="/lives/test.m3u8" type="application/x-mpegURL" />
                      </video>


                    <div class="liveButtons">
                      <div>Detector configured IP address: <b>{{env('DETECTOR_IP')}}</b></div>
                      <div class="liveControl">
                          <button type="button" class="btn btn-success" onclick=startDetector()> Start detector</button>
                          <button type="button" class="btn btn-danger" onclick=stopDetector()> Stop detector</button>
                          <button type="button" class="btn btn-primary" onclick=getData()> Fetch data</button>
                          <button type="button" class="btn btn-primary" onclick=changeData()> Change data</button>
                        </div>
                      <div>Configured source:</div> <input type="text" id="src">
                      <div>Needed frames for detection:</div> <input type="number" id="frames">
                      <div>Confidence treshold:</div> <input type="number" id="conf">
                      <div>Maximum detections per image:</div> <input type="number" id="max">
                      <div>Minimum needed distance between people:</div> <input type="number" id="min">
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
var player = videojs('my-video');
player.play();
</script>
