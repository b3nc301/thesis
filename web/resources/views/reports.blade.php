<head>
    <title>Video reports</title>
    <link rel="stylesheet"href="css/layout.css" >
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
                    </table>
                    </div>
                </div>
            </div>
        </div>
    </x-app-layout>
