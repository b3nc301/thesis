<head>
    <title>Video library</title>
    <link rel="stylesheet"href="css/layout.css" >
    </head>
    <x-app-layout>

        <x-slot name="header">
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">
                {{ __('Video Library') }}
            </h2>
        </x-slot>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                    <div class="p-6 bg-white border-b border-gray-200">
                      <div class="eventSearch">
                            <label for="starttime" >Start Time:</label>
                            <input type="datetime-local" id="starttime" name="starttime" class="m-2">
                            <label for="endtime" >End Time:</label>
                            <input type="datetime-local" id="endtime" name="endtime" class="m-2">
                            <button type="button" class="btn btn-primary">Search</button>
                      </div>

                    </div>

                </div>
                <table class="table">
                    <thead>
                      <tr>
                        <th scope="col">videoID</th>
                        <th scope="col">videoName</th>
                        <th scope="col">Date</th>
                        <th scope="col">Avalible?</th>
                        <th scope="col">View</th>
                        <th scope="col">Save</th>
                        <th scope="col">Delete</th>
                      </tr>
                    </thead>
                    <tbody>

                        @foreach ($videos as $video)
                        <tr>
                        <th scope="col">{{$video->id}}</th>
                        <td>{{$video->videoName}}</td>
                        <td>{{$video->videoDate}}</td>
                        <td>{{$video->videoAvailable}}</td>
                        <td><button type="button" class="btn btn-success" data-bs-toggle="modal" data-bs-target="#viewModal" data-bs-url={{$video->videoURL}}> View</button></td>
                        <td><a href="{{$video->videoURL}}" download class="btn btn-primary">Download</a></td>
                        <td><button type="button" class="btn btn-danger" data-bs-toggle="modal" data-bs-target="#deleteModal" data-bs-id={{$video->id}} data-bs-name={{$video->videoName}}>Delete</button></td>
                    </tr>
                        @endforeach
                    </tbody>
                  </table>
                  <!--view modal-->
                  <div class="modal fade" id="viewModal" tabindex="-1" aria-labelledby="viewModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-xl">
                      <div class="modal-content">
                        <div class="modal-header">
                          <h5 class="modal-title" id="viewModalLabel">View Video</h5>
                          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">

                        </div>
                        <div class="modal-footer">
                          <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                      </div>
                    </div>
                  </div>
                  <!--delete modal-->
                  <div class="modal fade" id="deleteModal" tabindex="-1" aria-labelledby="deleteModalLabel" aria-hidden="true">
                    <div class="modal-dialog">
                      <div class="modal-content">
                        <div class="modal-header">
                          <h5 class="modal-title" id="viewModalLabel">Delete video</h5>
                          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            Are you sure want to delete <span id="del" name="del"></span> ?
                        </div>
                        <div class="modal-footer">
                          <button type="button" class="btn btn-danger" id="null" name="deletebutton" onclick="deleteVid(this)">Delete</button>
                          <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                      </div>
                    </div>
                  </div>
            </div>
        </div>
    </x-app-layout>
