# pet-detector


Left off here:

This is my code:
    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
        
            if self.capture is None:
                self.spin(2)
                continue
            if self.capture.isOpened() and self.online:
                # Read next frame from stream and insert into deque
                status, frame = self.capture.read()
                if status:

                    #Process every frame_skip_interval frame
                    if frame_count % frame_skip_interval != 0:
                        continue

                    
                    # Check if the frame is valid
                    if frame is None or frame.size == 0:
                        logging.warning("Received an empty frame.")
                        continue

                    frame_bytes = None
                    # Convert the frame to bytes (e.g., using JPEG encoding)
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        logging.info("Frame encoded successfully.")
                    except Exception as e:
                        logging.error(f"Error encoding frame: {e}")
                        continue
                        
                    if frame_bytes is not None:
                                                
                        self.submit_attestation(frame_bytes, timeout=10)

                        detected_people = detect_people(frame)

                    for person in detected_people:
                        bbox = person['bbox']
                        ymin, xmin, ymax, xmax = bbox
                        h, w, _ = frame.shape
                        xmin = int(xmin * w)
                        xmax = int(xmax * w)
                        ymin = int(ymin * h)
                        ymax = int(ymax * h)
                        label = person['class'] + ":" + str(person['score'])
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.deque.append(frame)

                    # Automatically skip to the next frame
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()
                    self.capture.grab()

                else:
                    self.capture.release()
                    self.online = False
            else:
                # Attempt to reconnect
                print('attempting to reconnect', self.camera_stream_link)
                self.load_network_stream()
                self.spin(2)
            self.spin(.001)

    def submit_attestation(self, frame_bytes, timeout):
        """Submit the attestation for a single frame."""
        # Create a SHA256 hash of the frame bytes
        frame_hash = hashlib.sha256(frame_bytes).digest()
        
        # Convert the frame hash to a hexadecimal string
        frame_hash_hex = frame_hash.hex()
                        
        # Create a PendingAttestation for the frame
        attestation = PendingAttestation(frame_hash.hex())
        
        # Serialize the attestation
        ctx = BytesSerializationContext()
        attestation.serialize(ctx)

        # Save the serialized attestation to a file
        attestation_file_path = os.path.join(self.output_dir, f"attestation_{frame_hash_hex}.ots")
        with open(attestation_file_path, 'wb') as f:
            f.write(ctx.getbytes())
        logging.info(f"Attestation saved to {attestation_file_path}")

        # Submit the attestation to each calendar URL
        q = Queue()
        submit_async(self.notary_url, ctx.getbytes(), q, timeout)

        # Handle response
        try:
            result = q.get(block=True, timeout=timeout)
            if isinstance(result, Timestamp):
                logging.info("Attestation for frame submitted successfully.")
            else:
                logging.warning(f"Submission failed for frame: {result}")
        except Empty:
            logging.warning("Submission timed out for frame.")

With a log output of:
2025-04-07 16:05:31,821 - INFO - Frame encoded successfully.
2025-04-07 16:05:31,821 - INFO - Attestation saved to detected/attestation_127e0cf4207838243832631839646865353f46761b330b52b7314084d77e6abb.ots
2025-04-07 16:05:31,821 - INFO - Submitting to remote calendar https://a.pool.opentimestamps.org
2025-04-07 16:05:32,089 - WARNING - Submission failed for frame: HTTP Error 400: Bad Request


alt version:

                        frame_bytes = None
                        # Convert the frame to bytes (e.g., using JPEG encoding)
                        try:
                            _, buffer = cv2.imencode('.jpg', frame)
                            frame_bytes = buffer.tobytes()
                            logging.info("Frame encoded successfully.")
                        except Exception as e:
                            logging.error(f"Error encoding frame: {e}")
                            continue
                            
                        if frame_bytes is not None:
                                                    
                            # Create a SHA256 hash of the frame bytes
                            frame_hash = hashlib.sha256(frame_bytes).digest()

                            # Create a timestamp for the frame
                            timestamp = Timestamp(frame_hash)

                            # Create an attestation for the frame
                            attestation = UnknownAttestation(frame_hash, b'Frame data for frame count: {}'.format(frame_count))

                            # Serialize the attestation
                            ctx = BytesSerializationContext()
                            attestation.serialize(ctx)

                            # Save the serialized attestation to a file
                            attestation_file_path = os.path.join(self.output_dir, f"attestation_{frame_count}.ots")
                            with open(attestation_file_path, 'wb') as f:
                                f.write(ctx.getbytes())
                            logging.info(f"Attestation saved to {attestation_file_path}")

                            detected_people = detect_people(frame)



OR:

                        frame_bytes = None
                        # Convert the frame to bytes (e.g., using JPEG encoding)
                        try:
                            _, buffer = cv2.imencode('.jpg', frame)
                            frame_bytes = buffer.tobytes()
                            logging.info("Frame encoded successfully.")
                        except Exception as e:
                            logging.error(f"Error encoding frame: {e}")
                            continue

                        if frame_bytes is not None:
                            # Create a SHA256 hash of the frame bytes
                            frame_hash = hashlib.sha256(frame_bytes).digest()

                            # Create a timestamp proof for the frame hash
                            timestamp = Timestamp(frame_hash)

                            # Log the created timestamp for debugging
                            logging.info(f"Created Timestamp: {timestamp}")

                            # Check if the timestamp already exists in the cache
                            if frame_hash in self.cache:
                                existing_timestamp = self.cache[frame_hash]
                                logging.debug(f"Existing Timestamp before merge: {existing_timestamp}")
                                existing_timestamp.merge(timestamp)  # Merge with existing timestamp
                                logging.info("Merged with existing timestamp.")
                            else:
                                logging.info("Creating nonced version of the timestamp.")
                                logging.debug(f"Timestamp before noncing: {timestamp}")

                                try:
                                    nonced_timestamp = nonce_timestamp(timestamp)
                                    logging.info("Nonced version created successfully.")
                                    logging.debug(f"Nonced Timestamp: {nonced_timestamp}")
                                    logging.debug(f"Nonced Timestamp Operations: {nonced_timestamp.ops}")
                                    logging.debug(f"Nonced Timestamp Attestations: {nonced_timestamp.attestations}")
                                except Exception as e:
                                    logging.error(f"Error creating nonced timestamp: {e}")
                                    continue

                                # Check if the nonced timestamp is valid
                                if nonced_timestamp is None or not nonced_timestamp.ops:
                                    logging.error("Nonced timestamp is invalid or empty.")
                                    continue

                                if not nonced_timestamp.ops and not nonced_timestamp.attestations:
                                    logging.warning("Attempting to merge an empty nonced timestamp.")
                                    continue
                                
                                # Save the timestamp to the cache (this will also call save internally)
                                logging.info("Saving the nonced timestamp to the cache.")
                                self.cache.merge(nonced_timestamp)

                                # Create a temporary file for serialization
                                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                                    serialization_context = StreamSerializationContext(temp_file)

                                    try:
                                        logging.info("Serializing the nonced timestamp.")
                                        nonced_timestamp.serialize(serialization_context)
                                        logging.info("Serialization successful.")
                                    except Exception as e:
                                        logging.error(f"Error during serialization: {e}")
                                        continue  # Skip to the next iteration if there's an error

                                    # Get the serialized bytes from the temporary file
                                    temp_file.seek(0)  # Move to the beginning of the file
                                    serialized_timestamp = temp_file.read()

                                # Save the serialized timestamp to an .ots file
                                timestamp_file_path = os.path.join(self.output_dir, f"timestamp_{self.frame_count}.ots")
                                try:
                                    logging.info(f"Attempting to save timestamp to {timestamp_file_path}")
                                    with open(timestamp_file_path, 'wb') as f:
                                        f.write(serialized_timestamp)
                                    # Log the saved timestamp file path
                                    logging.info(f"Nonced Timestamp saved to {timestamp_file_path}")
                                except IOError as exp:
                                    logging.error(f"Failed to create timestamp file {timestamp_file_path}: {exp}")

                        detected_people = detect_people(frame)