<head>
    <title>Live video</title>
<style>
    .button {
  border: none;
  color: white;
  padding: 10px 10px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
}
    .button1 {background-color: #4CAF50;} /* Green */

</style>
</head>
    <x-app-layout>

        <x-slot name="header">
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">
                {{ __('Dashboard') }}
            </h2>
        </x-slot>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                    <div class="p-6 bg-white border-b border-gray-200">
                       <!-- <video id="video-help" width="530" controls>
                            <source id="videoPath" src="whatever.php?video=loaded.mp4" type="video/mp4">
                      </video>
                    -->
                      <div>Video settings:</div>
                      <div>Detector configured IP address: <b>{{env('DETECTOR_IP')}}</b></div>
                      <div>Configured source:  <button type="button" class="button button1"> Modify</button></div>
                      <div>Configured weights: <button type="button" class="button button1"> Modify</button></div>
                    </div>
                </div>
            </div>
        </div>
    </x-app-layout>
