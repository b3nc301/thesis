<head>
    <title>Live video</title>
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
                    <video width="1920" height="1080" class="mb-5">
                        <source src="'+document.location.origin+'/'+url+'#t=2,4" type="video/webm" preload="metadata">Your browser does not support the video tag.
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
    <script src="js/socket.js"></script>
