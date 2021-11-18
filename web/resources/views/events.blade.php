<head>
    <title>Events reports</title>
    </head>
    <x-app-layout>

        <x-slot name="header">
            <h2 class="font-semibold text-xl text-gray-800 leading-tight">
                {{ __('Reports') }}
            </h2>
        </x-slot>

        <div class="py-12">
            <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                    <div class="p-6 bg-white border-b border-gray-200">
                        <form action="/events" method="GET">
                            @csrf
                        <div class="eventSearch">
                            <label for="starttime" >Start Time:</label>
                            <input type="datetime-local" id="starttime" name="starttime" class="m-2" step="1" value=<?php if(isset($_GET['starttime'])) echo( $_GET['starttime']) ?>>
                            <label for="endtime" >End Time:</label>
                            <input type="datetime-local" id="endtime" name="endtime" class="m-2" step="1" value=<?php if(isset($_GET['endtime'])) echo( $_GET['endtime']) ?>>
                            <label for="level">Event alarm level:</label>
                            <select class="form-select-sm m-2" id="level" name="level" placeholder="Choose a level..." ">
                                <option>Not selected</option>
                                <option value="1" @if(isset($_GET['level']) && $_GET['level'] == '1')selected @endif>1</option>
                                <option value="2" @if(isset($_GET['level']) && $_GET['level'] == '2')selected @endif>2</option>
                                <option value="3" @if(isset($_GET['level']) && $_GET['level'] == '3')selected @endif>3</option>
                                <option value="4" @if(isset($_GET['level']) && $_GET['level'] == '4')selected @endif>4</option>
                                </select>
                            <button type="submit" class="btn btn-primary">Search</button>
                      </div>
                    </form>
                    </div>

                </div>
                <div class="table-responsive">
                <table class="table">
                    <thead>
                      <tr>
                        <th scope="col">eventID</th>
                        <th scope="col">Event type</th>
                        <th scope="col">Time</th>
                        <th scope="col">Duration</th>
                        <th scope="col">Video name</th>
                        <th scope="col">Event level</th>
                        <th scope="col">View</th>
                        <th scope="col">Save</th>
                        <th scope="col">Delete</th>
                      </tr>
                    </thead>
                    <tbody>
                        @foreach ($events as $report)
                        <tr>
                        <th scope="col">{{$report->id}}</th>
                        <td>{{$report->classID}}</td>
                        <td>{{$report->time}}</td>
                        <td>{{$report->videoframe/60}}</td>
                        <td>asd</td>
                        <td>{{$report->level}}</td>
                        <td><button type="button" class="btn btn-success" data-bs-toggle="modal" data-bs-target="#viewModal" data-bs-url={{$report->video->videoURL}}> View</button></td>
                        <td><a href={{$report->video->videoURL}} download class="btn btn-primary">Download</a></td>
                        <td><button type="button" class="btn btn-danger" data-bs-toggle="modal" data-bs-target="#deleteModal" data-bs-id={{$report->id}} data-bs-name={{$report->video->videoName}}>Delete</button></td>
                    </tr>
                        @endforeach
                    </tbody>
                  </table>
                </div>
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
