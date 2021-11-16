<head>
    <title>Video reports</title>
    <link rel="stylesheet"href="css/layout.css" >
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
                      <div class="eventSearch">
                            <label for="starttime" >Start Time:</label>
                            <input type="datetime-local" id="starttime" name="starttime" class="m-2">
                            <label for="endtime" >End Time:</label>
                            <input type="datetime-local" id="endtime" name="endtime" class="m-2">
                            <label for="level">Event alarm level:</label>
                            <select class="form-select-sm m-2" id="level" name="level" placeholder="Choose a level...">
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="3">3</option>
                                <option value="4">4</option>
                                </select>
                            <button type="button" class="btn btn-primary">Search</button>
                      </div>

                    </div>

                </div>
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
                        <td><button type="button" class="btn btn-success">View</button></td>
                        <td><button type="button" class="btn btn-primary">Download</button></td>
                        <td><button type="button" class="btn btn-danger">Delete</button></td>
                    </tr>
                        @endforeach
                    </tbody>
                  </table>
            </div>
        </div>
    </x-app-layout>
