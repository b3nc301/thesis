var viewModal = document.getElementById('viewModal')
var deleteModal = document.getElementById('deleteModal')
viewModal.addEventListener('show.bs.modal', function (event) {
  // Button that triggered the modal
  var button = event.relatedTarget
  // Extract info from data-bs-* attributes
  var url = button.getAttribute('data-bs-url')
  // If necessary, you could initiate an AJAX request here
  // and then do the updating in a callback.
  //
  // Update the modal's content.

  //var modalVideoView = viewModal.querySelector('.modal-body source')
  var modalVideoView = viewModal.querySelector('.modal-body')

    modalVideoView.innerHTML = '<video width="1920" height="1080" controls><source src="'+document.location.origin+'/'+url+'" type="video/webm" preload="metadata">Your browser does not support the video tag.</video>'
  //modalVideoView.setAttribute("src", document.location.origin+"/"+url)
})
deleteModal.addEventListener('show.bs.modal', function (event) {
    // Button that triggered the modal
    var button = event.relatedTarget
    // Extract info from data-bs-* attributes
    var name = button.getAttribute('data-bs-name')
    // If necessary, you could initiate an AJAX request here
    // and then do the updating in a callback.
    //
    // Update the modal's content.

    //var modalVideoView = viewModal.querySelector('.modal-body source')
    var modalVideoDeleteButton = deleteModal.querySelector('.modal-footer button[name="deletebutton"]')
    var modalVideoDelete = deleteModal.querySelector('.modal-body span')



    modalVideoDelete.innerHTML = name;
    modalVideoDeleteButton.id=button.getAttribute('data-bs-id');
    //modalVideoView.setAttribute("src", document.location.origin+"/"+url)
  })

function deleteVid(tag){
    console.log(tag.id);


        let data = {id: tag.id}

        fetch('/deleteVideo', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-TOKEN': document.getElementsByName("csrf-token")[0].getAttribute('content')

        },
        body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            location.reload();
        })
        .catch((error) => {
        console.error('Error:', error);
        });
}
