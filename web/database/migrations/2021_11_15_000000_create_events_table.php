<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateEventsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('events', function (Blueprint $table) {
            $table->id();
            $table->float("xbox");
            $table->float("ybox");
            $table->float("wbox");
            $table->float("hbox");
            $table->integer("classID");
            $table->timestamp("time");
            $table->integer("videoTime");
            $table->integer("videoID");
            $table->integer("level");
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('CreateEventsTable');
    }
}
