<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Videos extends Model
{
    use HasFactory;

    protected $table = 'videos';
    protected $primaryKey = 'id';
    public $timestamps = false;

    protected $fillable = [
		'videoName',
		'videoDate',
		'videoURL',
		'videoAvailable',
	];
    protected $casts = [
        'videoDate' => 'datetime',
        'videoAvailable' => 'boolean'
    ];

    public function events(){

	}

}
